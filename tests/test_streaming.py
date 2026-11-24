"""SSE streaming tests — the stream must apply the same output guard as /api/travel."""

import json
from types import SimpleNamespace
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

    async def aget_state(self, _config):
        return SimpleNamespace(next=(), interrupts=())


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
        {"event": "on_chain_start", "name": "flight_agent", "data": {}},
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

    progress = [(d["agent"], d["status"]) for e, d in parsed if e == "progress"]
    assert progress == [("flight_agent", "started"), ("flight_agent", "completed")]

    by_event = dict(parsed)
    assert "[REDACTED:GROQ_KEY]" in by_event["agent_result"]["flight_results"]
    complete = by_event["complete"]
    # A fresh thread per run — reducer-backed state must not leak between requests
    assert complete["thread_id"].startswith("user_")
    assert complete["thread_id"] != "t1"
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


# ── Live answer streaming ─────────────────────────────────────────────

class TestAnswerStreamer:
    def test_buffers_until_a_line_is_complete(self):
        streamer = streaming.AnswerStreamer()
        assert streamer.feed("## Trip ") is None
        assert streamer.feed("Summary") is None
        assert streamer.started is False
        assert streamer.feed("\n\nDay 1: Shin") == "## Trip Summary\n\n"
        assert streamer.started is True
        assert streamer.feed("juku\n") == "Day 1: Shinjuku\n"
        assert streamer.flush() is None

    def test_flush_returns_last_line_without_newline(self):
        streamer = streaming.AnswerStreamer()
        streamer.feed("line one\nlast line")
        assert streamer.flush() == "last line"
        assert streamer.flush() is None

    def test_secret_split_across_tokens_is_redacted(self):
        """Token-level guarding would miss this; line-level catches it."""
        streamer = streaming.AnswerStreamer()
        half = len(FAKE_KEY) // 2
        assert streamer.feed("key: " + FAKE_KEY[:half]) is None
        out = streamer.feed(FAKE_KEY[half:] + "\n")
        assert FAKE_KEY not in out
        assert out == "key: [REDACTED:GROQ_KEY]\n"

    def test_chunk_text_handles_block_lists_and_garbage(self):
        assert streaming._chunk_text(AIMessage(content="abc")) == "abc"
        assert streaming._chunk_text(SimpleNamespace(content=[{"type": "text", "text": "a"}, "b"])) == "ab"
        assert streaming._chunk_text(None) == ""


def _token(text, node="orchestrator"):
    return {"event": "on_chat_model_stream", "name": "ChatGroq",
            "metadata": {"langgraph_node": node}, "data": {"chunk": AIMessage(content=text)}}


@pytest.mark.asyncio
async def test_stream_emits_answer_deltas_only_for_the_final_answer():
    events = [
        _token("slot extraction\n", node="intent_classifier"),     # other nodes' tokens stay private
        _token("search notes\n", node="flight_agent"),
        {"event": "on_chain_start", "name": "orchestrator", "data": {}},
        _token("## Plan\n"),
        _token("Day 1: To"),
        _token("kyo\nDay 2: Kyoto"),
        {"event": "on_chain_end", "name": "orchestrator",
         "data": {"output": {"messages": [AIMessage(content="## Plan\nDay 1: Tokyo\nDay 2: Kyoto")]}}},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {
            "messages": [AIMessage(content="## Plan\nDay 1: Tokyo\nDay 2: Kyoto")],
        }}},
    ]
    parsed = await _collect(events)
    deltas = [d["text"] for e, d in parsed if e == "answer_delta"]
    assert deltas == ["## Plan\n", "Day 1: Tokyo\n", "Day 2: Kyoto"]
    assert "".join(deltas) == dict(parsed)["complete"]["answer"]

    # The tail is flushed before the node is reported complete
    order = [(e, d.get("status")) for e, d in parsed if e in {"answer_delta", "progress"}]
    assert order[-2:] == [("answer_delta", None), ("progress", "completed")]


def _llm_start(node="orchestrator"):
    return {"event": "on_chat_model_start", "name": "ChatGroq", "metadata": {"langgraph_node": node}, "data": {}}


def _delegation(*names):
    calls = [{"name": n, "args": {}, "id": f"call_{n}", "type": "tool_call"} for n in names]
    return {"messages": [AIMessage(content="", tool_calls=calls)]}


@pytest.mark.asyncio
async def test_text_before_a_delegation_is_not_the_answer():
    """Only the orchestrator's last turn is the answer; earlier text is retracted or never sent."""
    events = [
        # Turn 1: one finished line and an unfinished one, then the turn ends in tool calls
        {"event": "on_chain_start", "name": "orchestrator", "data": {}},
        _llm_start(),
        _token("Let me look that up.\n"),
        _token("Checking the wea"),
        {"event": "on_chain_end", "name": "orchestrator", "data": {"output": _delegation("weather_agent")}},
        {"event": "on_chain_start", "name": "weather_agent", "data": {}},
        _llm_start(node="weather_agent"),                             # a sub-agent's LLM call resets nothing
        {"event": "on_chain_end", "name": "weather_agent", "data": {"output": {"weather_results": "sunny"}}},
        # Turn 2: the answer
        {"event": "on_chain_start", "name": "orchestrator", "data": {}},
        _llm_start(),
        _token("Sunny all week.\n"),
        _token("Pack light."),
        {"event": "on_chain_end", "name": "orchestrator",
         "data": {"output": {"messages": [AIMessage(content="Sunny all week.\nPack light.")]}}},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {
            "messages": [AIMessage(content="Sunny all week.\nPack light.")], "weather_results": "sunny",
        }}},
    ]
    parsed = await _collect(events)
    stream = [(e, d.get("text")) for e, d in parsed if e in {"answer_delta", "answer_reset"}]
    assert stream == [
        ("answer_delta", "Let me look that up.\n"),
        ("answer_reset", None),                # ...and the unfinished line is dropped silently
        ("answer_delta", "Sunny all week.\n"),
        ("answer_delta", "Pack light."),
    ]
    assert dict(parsed)["agent_result"] == {"weather_results": "sunny"}
    assert parsed[-1][0] == "complete"


@pytest.mark.asyncio
async def test_a_second_llm_call_in_one_turn_replaces_the_first():
    """A failed decision call followed by the tool-free answer call: only the latter counts."""
    events = [
        {"event": "on_chain_start", "name": "orchestrator", "data": {}},
        _llm_start(),
        _token("Half an ans\n"),
        _llm_start(),
        _token("The real answer."),
        {"event": "on_chain_end", "name": "orchestrator",
         "data": {"output": {"messages": [AIMessage(content="The real answer.")]}}},
        {"event": "on_chain_end", "name": "LangGraph",
         "data": {"output": {"messages": [AIMessage(content="The real answer.")]}}},
    ]
    parsed = await _collect(events)
    stream = [(e, d.get("text")) for e, d in parsed if e in {"answer_delta", "answer_reset"}]
    assert stream == [("answer_delta", "Half an ans\n"), ("answer_reset", None), ("answer_delta", "The real answer.")]


@pytest.mark.asyncio
async def test_no_reset_when_nothing_was_streamed():
    events = [
        {"event": "on_chain_start", "name": "orchestrator", "data": {}},
        _llm_start(),
        {"event": "on_chain_end", "name": "orchestrator", "data": {"output": _delegation("flight_agent")}},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"messages": [AIMessage(content="x")]}}},
    ]
    assert "answer_reset" not in [e for e, _ in await _collect(events)]
