"""Structured trip slots: one LLM call turns the raw query into typed fields.

Without this, every agent received only the raw sentence and had to guess the
destination, origin and dates on its own (flight/hotel agents literally used
the whole query as the "destination" in their search templates).

The same call also classifies intent, so slot extraction costs no extra LLM
round-trip compared with the previous LLM intent fallback.
"""

import json
import logging
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from navigo_agent.config import get_llm

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset({"flight", "hotel", "weather", "itinerary", "full_trip", "general"})
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "unknown", "not specified", "unspecified"}


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class TripSlots(BaseModel):
    """Trip parameters extracted from the user's request. Every field is optional."""

    origin: str | None = Field(default=None, description="Departure city/country")
    destination: str | None = Field(default=None, description="Destination as the user named it")
    destination_city: str | None = Field(
        default=None, description="A single city usable for a weather lookup"
    )
    start_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD")
    duration_days: int | None = None
    travelers: int | None = None
    budget: str | None = Field(default=None, description="Free text: 'budget', 'mid-range', '$2000'")
    interests: list[str] = Field(default_factory=list)

    @field_validator(
        "origin", "destination", "destination_city", "start_date", "end_date", "budget",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value):
        if value is None:
            return None
        text = str(value).strip()
        return None if text.lower() in _NULL_STRINGS else text

    @field_validator("duration_days", "travelers", mode="before")
    @classmethod
    def _clean_int(cls, value):
        try:
            number = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @field_validator("interests", mode="before")
    @classmethod
    def _clean_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip().lower() not in _NULL_STRINGS]

    @property
    def weather_city(self) -> str | None:
        """Best city to query the weather API with."""
        return self.destination_city or self.destination


EXTRACTION_SYSTEM_PROMPT = """You extract structured trip details from a travel request.

Today's date is {today}. Resolve relative dates ("next Friday", "this December") against it.

Return ONLY a JSON object with exactly these fields (use null when the user did not say):
{{
  "intent": "flight | hotel | weather | itinerary | full_trip | general",
  "origin": "<departure city or country, or null>",
  "destination": "<destination as the user named it (city, region or country), or null>",
  "destination_city": "<ONE city for a weather lookup: the destination itself if it is a city, otherwise its main tourist city / capital, or null>",
  "start_date": "<YYYY-MM-DD or null>",
  "end_date": "<YYYY-MM-DD or null>",
  "duration_days": <integer or null>,
  "travelers": <integer or null>,
  "budget": "<the user's budget wording, e.g. 'budget', 'mid-range', 'luxury', '$2000', or null>",
  "interests": ["<interest>", ...]
}}

Intent meanings:
- flight: only asking about flights, airlines, airports
- hotel: only asking about hotels / places to stay
- weather: only asking about weather, temperature, climate
- itinerary: asking for a day-by-day plan only
- full_trip: asking for a complete trip plan (flights, hotels, activities)
- general: travel-related but none of the above

Rules:
- Never invent values. If the user did not mention something, use null (or [] for interests).
- If only a duration and a start date are given, compute end_date; otherwise leave it null.
- Output ONLY the JSON. No markdown, no explanation."""


def parse_trip_extraction(raw: str) -> tuple[str | None, TripSlots]:
    """Parse the extractor's output. Never raises — returns (None, empty) on garbage."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        logger.warning("Trip extraction returned no JSON object: %.200s", raw)
        return None, TripSlots()

    try:
        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise TypeError("not a JSON object")
        intent = str(data.pop("intent", "") or "").strip().lower()
        known = {k: v for k, v in data.items() if k in TripSlots.model_fields}
        slots = TripSlots(**known)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse trip extraction: %s. Raw: %.200s", e, raw)
        return None, TripSlots()

    return (intent if intent in VALID_INTENTS else None), slots


async def extract_trip_request(query: str) -> tuple[str | None, TripSlots]:
    """Run the extraction LLM call. Returns (intent or None, slots); never raises."""
    try:
        llm = get_llm(temperature=0.0, max_tokens=400)
        response = await llm.ainvoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT.format(today=_today())),
            HumanMessage(content=query),
        ])
    except Exception as e:  # noqa: BLE001
        logger.warning("Trip extraction LLM call failed: %s", e)
        return None, TripSlots()

    return parse_trip_extraction(str(response.content))


def format_trip_details(trip: dict | None) -> str:
    """Render slots as a prompt block. Only known fields are listed."""
    slots = TripSlots(**(trip or {}))
    lines = [f"- Today's date: {_today()}"]
    labelled = [
        ("Origin", slots.origin),
        ("Destination", slots.destination),
        ("Start date", slots.start_date),
        ("End date", slots.end_date),
        ("Duration (days)", slots.duration_days),
        ("Travelers", slots.travelers),
        ("Budget", slots.budget),
        ("Interests", ", ".join(slots.interests) or None),
    ]
    lines += [f"- {label}: {value}" for label, value in labelled if value]
    missing = [label for label, value in labelled[:3] if not value]
    if missing:
        lines.append(
            f"- Not specified by the user: {', '.join(missing)} "
            "(do not invent these; give general guidance instead)"
        )
    return "\n".join(lines)
