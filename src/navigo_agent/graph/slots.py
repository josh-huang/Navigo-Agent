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


FOLLOW_UP_ADDENDUM = """

This message continues an existing conversation.

Previous trip details:
{previous_trip}

Conversation so far:
{history}

Additional rules for this message:
- Add one more field to the JSON: "is_follow_up": true or false.
  true  = the message refines, changes or asks more about the trip above
          ("make it cheaper", "add 2 days", "what about hotels near the station?").
  false = the message starts a new, unrelated request (a different trip).
- If is_follow_up is true: return the UPDATED trip details — keep every previous value the
  user did not change, and classify intent by what must be (re)done to answer:
  "full_trip" when the whole plan has to be revised, or a single topic when only that is asked.
- If is_follow_up is false: ignore the previous trip details completely."""


def _parse_extraction(raw: str) -> tuple[str | None, TripSlots, dict]:
    """Parse extractor output into (intent, slots, raw dict). Never raises."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        logger.warning("Trip extraction returned no JSON object: %.200s", raw)
        return None, TripSlots(), {}

    try:
        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise TypeError("not a JSON object")
        intent = str(data.get("intent", "") or "").strip().lower()
        known = {k: v for k, v in data.items() if k in TripSlots.model_fields}
        slots = TripSlots(**known)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Failed to parse trip extraction: %s. Raw: %.200s", e, raw)
        return None, TripSlots(), {}

    return (intent if intent in VALID_INTENTS else None), slots, data


def parse_trip_extraction(raw: str) -> tuple[str | None, TripSlots]:
    """Parse the extractor's output. Never raises — returns (None, empty) on garbage."""
    intent, slots, _ = _parse_extraction(raw)
    return intent, slots


async def _call_extractor(system_prompt: str, query: str) -> str | None:
    try:
        llm = get_llm(temperature=0.0, max_tokens=400)
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ])
    except Exception as e:  # noqa: BLE001
        logger.warning("Trip extraction LLM call failed: %s", e)
        return None
    return str(response.content)


async def extract_trip_request(query: str) -> tuple[str | None, TripSlots]:
    """Run the extraction LLM call. Returns (intent or None, slots); never raises."""
    raw = await _call_extractor(EXTRACTION_SYSTEM_PROMPT.format(today=_today()), query)
    return parse_trip_extraction(raw) if raw is not None else (None, TripSlots())


async def extract_follow_up(
    query: str,
    history: list[dict[str, str]],
    previous_trip: dict | None,
) -> tuple[str | None, TripSlots, bool]:
    """Extraction for a message that continues a conversation.

    Returns (intent or None, slots, is_follow_up). If the call or parsing
    fails, the message is treated as a follow-up with no new slot values, so
    the previous trip details are kept. Never raises.
    """
    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(today=_today()) + FOLLOW_UP_ADDENDUM.format(
        previous_trip=json.dumps(TripSlots(**(previous_trip or {})).model_dump(exclude_none=True)),
        history=format_history(history) or "(none)",
    )
    raw = await _call_extractor(system_prompt, query)
    if raw is None:
        return None, TripSlots(), True

    intent, slots, data = _parse_extraction(raw)
    flag = data.get("is_follow_up", True)
    if isinstance(flag, str):
        flag = flag.strip().lower() not in {"false", "no", "0"}
    return intent, slots, bool(flag)


# ── Conversation history ──────────────────────────────────────────────

MAX_HISTORY_TURNS = 4
MAX_STORED_ANSWER_CHARS = 2000


def append_turn(history: list[dict[str, str]] | None, user: str, assistant: str) -> list[dict[str, str]]:
    """Return history + this turn, trimmed to the most recent MAX_HISTORY_TURNS."""
    turn = {"user": user.strip(), "assistant": assistant.strip()[:MAX_STORED_ANSWER_CHARS]}
    return [*(history or []), turn][-MAX_HISTORY_TURNS:]


def format_history(history: list[dict[str, str]] | None, *, answer_chars: int = 300) -> str:
    """Render earlier turns for a prompt. Empty string when there is no history.

    Answers are cut to `answer_chars` — agents need the gist of what was said,
    the final synthesizer asks for more so it can revise the previous plan.
    """
    lines: list[str] = []
    for turn in history or []:
        answer = str(turn.get("assistant", ""))
        if len(answer) > answer_chars:
            answer = answer[:answer_chars].rstrip() + " …"
        lines.append(f"User: {turn.get('user', '')}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def conversation_block(history: list[dict[str, str]] | None, *, answer_chars: int = 300) -> str:
    """A ready-to-append prompt section about earlier turns ("" if none)."""
    text = format_history(history, answer_chars=answer_chars)
    if not text:
        return ""
    return (
        "\n\n## Conversation So Far\n\n"
        "The current request is a follow-up to this conversation. Read it in that light "
        "(\"it\", \"cheaper\", \"instead\" refer to what was discussed).\n\n" + text
    )


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


# ── Clarification: which slots must be known before agents are dispatched ──

SLOT_QUESTIONS = {
    "destination": "Where would you like to go?",
    "origin": "Which city are you departing from?",
    "duration": "How many days is the trip (or what are your travel dates)?",
}


def missing_required_slots(intent: str, trip: dict | None) -> list[str]:
    """Slots without which the given intent cannot be answered usefully.

    Deliberately minimal — only things an agent cannot reasonably assume:
      - any specific intent  → destination
      - flight queries       → origin as well
      - trip / itinerary     → trip length (days, or both dates) as well
    `general` questions never trigger a clarification.
    """
    if intent == "general":
        return []

    slots = TripSlots(**(trip or {}))
    missing: list[str] = []
    if not slots.destination:
        missing.append("destination")
    if intent in {"flight", "flights"} and not slots.origin:
        missing.append("origin")
    if intent in {"full_trip", "itinerary"} and not (
        slots.duration_days or (slots.start_date and slots.end_date)
    ):
        missing.append("duration")
    return missing


def build_clarification_question(missing: list[str]) -> str:
    """Turn the missing-slot list into one short question for the user."""
    questions = [SLOT_QUESTIONS[slot] for slot in missing if slot in SLOT_QUESTIONS]
    if len(questions) == 1:
        return f"Before I start planning, one quick question: {questions[0]}"
    bullets = "\n".join(f"- {q}" for q in questions)
    return f"Before I start planning, I need a few details:\n{bullets}"


def merge_slots(base: dict | None, update: TripSlots) -> dict:
    """Overlay newly extracted values on the existing slots (known values win only if new is empty)."""
    merged = TripSlots(**(base or {})).model_dump()
    for key, value in update.model_dump().items():
        if value not in (None, [], ""):
            merged[key] = value
    return merged
