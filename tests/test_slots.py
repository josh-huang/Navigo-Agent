"""Trip slot extraction tests (parsing + the intent_classifier node)."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

import navigo_agent.graph.routing as routing_mod
import navigo_agent.graph.slots as slots_mod
from navigo_agent.graph.slots import (
    TripSlots,
    extract_trip_request,
    format_trip_details,
    parse_trip_extraction,
)
from navigo_agent.state import build_initial_state

FULL = {
    "intent": "full_trip",
    "origin": "Dhaka",
    "destination": "Japan",
    "destination_city": "Tokyo",
    "start_date": "2026-12-01",
    "end_date": "2026-12-08",
    "duration_days": 7,
    "travelers": 2,
    "budget": "mid-range",
    "interests": ["food", "temples"],
}


class TestParseTripExtraction:
    def test_full_payload(self):
        intent, slots = parse_trip_extraction(json.dumps(FULL))
        assert intent == "full_trip"
        assert slots.origin == "Dhaka"
        assert slots.destination == "Japan"
        assert slots.weather_city == "Tokyo"
        assert slots.duration_days == 7
        assert slots.interests == ["food", "temples"]

    def test_code_fence_and_surrounding_text(self):
        raw = f"Here you go:\n```json\n{json.dumps(FULL)}\n```"
        intent, slots = parse_trip_extraction(raw)
        assert intent == "full_trip"
        assert slots.destination == "Japan"

    def test_null_like_values_are_normalised(self):
        raw = json.dumps({
            "intent": "weather", "origin": "null", "destination": " Paris ",
            "destination_city": "", "start_date": "N/A", "duration_days": "unknown",
            "travelers": "2", "budget": None, "interests": None,
        })
        intent, slots = parse_trip_extraction(raw)
        assert intent == "weather"
        assert slots.origin is None
        assert slots.destination == "Paris"
        assert slots.weather_city == "Paris"  # falls back to destination
        assert slots.start_date is None
        assert slots.duration_days is None
        assert slots.travelers == 2
        assert slots.interests == []

    def test_unknown_intent_and_extra_keys(self):
        intent, slots = parse_trip_extraction('{"intent": "banana", "destination": "Rome", "mood": "x"}')
        assert intent is None
        assert slots.destination == "Rome"

    @pytest.mark.parametrize("raw", ["", "no json here", "{not json}", "[1, 2]", '{"a": '])
    def test_garbage_never_raises(self, raw):
        intent, slots = parse_trip_extraction(raw)
        assert intent is None
        assert slots == TripSlots()


class TestExtractTripRequest:
    @pytest.mark.asyncio
    async def test_prompt_contains_today_and_query(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content=json.dumps(FULL))
        with (
            patch.object(slots_mod, "get_llm", lambda **_: llm),
            patch.object(slots_mod, "_today", lambda: "2026-11-11"),
        ):
            intent, slots = await extract_trip_request("Plan a week in Japan from Dhaka")
        system, human = llm.ainvoke.await_args.args[0]
        assert "Today's date is 2026-11-11" in system.content
        assert human.content == "Plan a week in Japan from Dhaka"
        assert intent == "full_trip"
        assert slots.origin == "Dhaka"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("rate limited")
        with patch.object(slots_mod, "get_llm", lambda **_: llm):
            assert await extract_trip_request("anything") == (None, TripSlots())


class TestFormatTripDetails:
    def test_lists_known_fields_only(self):
        text = format_trip_details({"origin": "Dhaka", "destination": "Tokyo", "duration_days": 5})
        assert "- Origin: Dhaka" in text
        assert "- Destination: Tokyo" in text
        assert "- Duration (days): 5" in text
        assert "Budget" not in text
        assert "Not specified by the user: Start date" in text

    def test_empty_trip(self):
        text = format_trip_details(None)
        assert "Today's date" in text
        assert "Not specified by the user: Origin, Destination, Start date" in text


class TestIntentClassifierNode:
    async def _run(self, query, extraction):
        with patch.object(routing_mod, "extract_trip_request", AsyncMock(return_value=extraction)):
            return await routing_mod.intent_classifier(build_initial_state(query))

    @pytest.mark.asyncio
    async def test_fast_path_intent_wins_but_slots_still_extracted(self):
        update = await self._run(
            "What is the weather in Tokyo right now?",
            ("general", TripSlots(destination="Tokyo")),
        )
        assert update["user_intent"] == "weather"
        assert update["trip"]["destination"] == "Tokyo"
        assert update["extracted_destination"] == "Tokyo"
        assert update["llm_calls"] == 1

    @pytest.mark.asyncio
    async def test_llm_intent_used_when_fast_path_unsure(self):
        update = await self._run("Tell me about Tokyo", ("general", TripSlots(destination="Tokyo")))
        assert update["user_intent"] == "general"

    @pytest.mark.asyncio
    async def test_extraction_failure_defaults_to_full_trip(self):
        update = await self._run("Tell me about Tokyo", (None, TripSlots()))
        assert update["user_intent"] == "full_trip"
        assert update["extracted_destination"] is None
        assert update["trip"]["destination"] is None
