"""SSE streaming tests — the stream must apply the same output guard as /api/travel."""

import json
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from navigo_agent import streaming

FAKE_KEY = "gsk_" + "a" * 40


class FakeGraph:
    def __init__(self, events):
        self._events = events

    async def astream_events(self, _state, config=None, version=None):
        for event in self._events:
            yield event


def _parse_sse(chunks: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for chunk in chunks:
        lines = chunk.strip().split("\n")
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return parsed


async def _collect(events):
    with patch.object(streaming, "get_compiled_graph", lambda: FakeGraph(events)):
        return _parse_sse([chunk async for chunk in streaming.stream_travel_plan("Plan a trip", "t1")])


@pytest.mark.asyncio
async def test_stream_redacts_and_sanitises_output():
    events = [
        {"event": "on_chain_end", "name": "flight_agent",
         "data": {"output": {"flight_results": f"key {FAKE_KEY} <script>alert(1)</script>"}}},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {
            "messages": [AIMessage(content=f"Plan {FAKE_KEY} <script>alert(1)</script>done")],
            "flight_results": f"key {FAKE_KEY}",
            "hotel_results": "<iframe src='x'></iframe>Hotel",
            "weather_results": "sunny",
            "itinerary": "Day 1",
            "llm_calls": 3,
        }}},
    ]
    parsed = await _collect(events)
    raw = json.dumps(parsed)
    assert FAKE_KEY not in raw
    assert "<script" not in raw
    assert "<iframe" not in raw

    by_event = dict(parsed)
    assert "[REDACTED:GROQ_KEY]" in by_event["agent_result"]["flight_results"]
    complete = by_event["complete"]
    assert complete["thread_id"] == "t1"
    assert complete["answer"].startswith("Plan [REDACTED:GROQ_KEY]")
    assert complete["hotel_results"] == "Hotel"
    assert complete["weather_results"] == "sunny"
    assert complete["llm_calls"] == 3


@pytest.mark.asyncio
async def test_stream_error_event_hides_details():
    class Boom:
        def astream_events(self, *_a, **_k):
            raise RuntimeError("SECRET_DB_PASSWORD=xyz")

    with patch.object(streaming, "get_compiled_graph", lambda: Boom()):
        chunks = [c async for c in streaming.stream_travel_plan("Plan a trip", "t1")]
    (event, data), = _parse_sse(chunks)
    assert event == "error"
    assert "SECRET_DB_PASSWORD" not in data["message"]
