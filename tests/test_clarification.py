"""Clarification (human-in-the-loop) tests: pause on missing info, resume with the reply."""

import itertools
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import navigo_agent
import navigo_agent.graph.agents.final as final_mod
import navigo_agent.graph.agents.flight as flight_mod
import navigo_agent.graph.agents.hotel as hotel_mod
import navigo_agent.graph.agents.itinerary as itinerary_mod
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.builder as builder_mod
import navigo_agent.graph.routing as routing_mod
import navigo_agent.graph.supervisor as supervisor_mod
from navigo_agent import streaming
from navigo_agent.graph.builder import build_travel_graph
from navigo_agent.graph.slots import (
    TripSlots,
    build_clarification_question,
    merge_slots,
    missing_required_slots,
)


class TestMissingRequiredSlots:
    @pytest.mark.parametrize(
        "intent,trip,expected",
        [
            ("weather", {}, ["destination"]),
            ("weather", {"destination": "Tokyo"}, []),
            ("hotel", {"destination": "Bangkok"}, []),
            ("flights", {"destination": "Dubai"}, ["origin"]),
            ("flight", {}, ["destination", "origin"]),
            ("flights", {"origin": "Dhaka", "destination": "Dubai"}, []),
            ("full_trip", {"destination": "Japan"}, ["duration"]),
            ("full_trip", {"destination": "Japan", "duration_days": 7}, []),
            ("full_trip", {"destination": "Japan", "start_date": "2026-12-01", "end_date": "2026-12-08"}, []),
            ("full_trip", {"destination": "Japan", "start_date": "2026-12-01"}, ["duration"]),
            ("itinerary", {}, ["destination", "duration"]),
            # full trips do not insist on an origin — too many valid requests omit it
            ("full_trip", {"destination": "Japan", "duration_days": 7, "origin": None}, []),
            ("general", {}, []),
            ("general", None, []),
        ],
    )
    def test_matrix(self, intent, trip, expected):
        assert missing_required_slots(intent, trip) == expected

    def test_single_question(self):
        q = build_clarification_question(["destination"])
        assert "Where would you like to go?" in q
        assert "\n" not in q

    def test_multiple_questions_are_bulleted(self):
        q = build_clarification_question(["destination", "duration"])
        assert q.count("\n- ") == 2

    def test_merge_keeps_known_values(self):
        merged = merge_slots(
            {"origin": "Dhaka", "budget": "mid-range", "interests": ["food"]},
            TripSlots(destination="Tokyo", duration_days=5),
        )
        assert merged["origin"] == "Dhaka"
        assert merged["budget"] == "mid-range"
        assert merged["interests"] == ["food"]
        assert merged["destination"] == "Tokyo"
        assert merged["duration_days"] == 5


# ── End-to-end through the real graph ─────────────────────────────────

class _SupervisorLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        agents = ["flight_agent", "hotel_agent", "weather_agent"] if self.calls == 1 else []
        return AIMessage(content=json.dumps({
            "reasoning": "scripted", "next_action": "x",
            "agents_to_dispatch": agents, "is_complete": not agents,
        }))


class _FinalLLM:
    async def ainvoke(self, _messages):
        return AIMessage(content="FINAL PLAN")


def _fake_react(answer):
    async def _run(**_kwargs):
        return answer, 1, []

    return _run


@pytest.fixture
def graph_env():
    """Real graph + in-memory checkpointer; LLMs/tools mocked.

    `extract` is the trip extractor mock — set its side_effect per test.
    """
    extract = AsyncMock()
    weather = AsyncMock(return_value="sunny")
    saved = builder_mod._compiled_graph
    with (
        patch.object(routing_mod, "extract_trip_request", extract),
        patch.object(supervisor_mod, "get_llm", lambda **_: _SupervisorLLM()),
        patch.object(final_mod, "get_llm", lambda **_: _FinalLLM()),
        patch.object(flight_mod, "get_llm", lambda **_: None),
        patch.object(hotel_mod, "get_llm", lambda **_: None),
        patch.object(itinerary_mod, "get_llm", lambda **_: None),
        patch.object(flight_mod, "run_react_loop", _fake_react("FLIGHTS")),
        patch.object(hotel_mod, "run_react_loop", _fake_react("HOTELS")),
        patch.object(itinerary_mod, "run_react_loop", _fake_react("ITINERARY")),
        patch.object(weather_mod, "weather_mcp_search", weather),
        patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="mild")),
    ):
        graph = build_travel_graph().compile(checkpointer=MemorySaver())
        builder_mod._compiled_graph = graph
        with patch.object(streaming, "get_compiled_graph", lambda: graph):
            try:
                yield {"graph": graph, "extract": extract, "weather": weather}
            finally:
                builder_mod._compiled_graph = saved


def _parse_sse(chunks):
    out = []
    for chunk in chunks:
        lines = chunk.strip().split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


class TestClarificationFlow:
    @pytest.mark.asyncio
    async def test_complete_request_is_not_interrupted(self, graph_env):
        graph_env["extract"].return_value = ("full_trip", TripSlots(destination="Tokyo", duration_days=5))
        result = await navigo_agent.run_travel_agent("Plan a 5-day trip to Tokyo")
        assert result["needs_clarification"] is False
        assert result["answer"] == "FINAL PLAN"
        assert graph_env["extract"].await_count == 1

    @pytest.mark.asyncio
    async def test_pause_then_resume(self, graph_env):
        graph_env["extract"].side_effect = [
            ("full_trip", TripSlots(origin="Dhaka", budget="mid-range")),          # first message
            ("full_trip", TripSlots(destination="Japan", destination_city="Tokyo", duration_days=7)),  # reply
        ]

        first = await navigo_agent.run_travel_agent("Plan a trip for me from Dhaka, mid-range budget")
        assert first["needs_clarification"] is True
        assert first["missing"] == ["destination", "duration"]
        assert "Where would you like to go?" in first["question"]
        assert first["answer"] == first["question"]
        assert first["flight_results"] == ""
        graph_env["weather"].assert_not_awaited()  # no agent ran before the user answered

        second = await navigo_agent.run_travel_agent("Japan, for a week", thread_id=first["thread_id"])
        assert second["needs_clarification"] is False
        assert second["thread_id"] == first["thread_id"]  # same run, resumed
        assert second["answer"] == "FINAL PLAN"
        assert second["itinerary"] == "ITINERARY"

        # The reply was merged with what was already known
        state = (await graph_env["graph"].aget_state(
            {"configurable": {"thread_id": first["thread_id"]}}
        )).values
        assert state["trip"]["origin"] == "Dhaka"
        assert state["trip"]["budget"] == "mid-range"
        assert state["trip"]["destination"] == "Japan"
        assert state["trip"]["duration_days"] == 7
        assert "Additional details from the user: Japan, for a week" in state["user_query"]
        graph_env["weather"].assert_awaited_once_with("Tokyo")
        # Re-extraction saw the original request AND the reply
        assert "from Dhaka" in graph_env["extract"].await_args_list[1].args[0]

    @pytest.mark.asyncio
    async def test_asks_only_once_even_if_reply_is_unhelpful(self, graph_env):
        graph_env["extract"].side_effect = [
            ("weather", TripSlots()),
            ("weather", TripSlots()),  # reply still has no destination
        ]
        first = await navigo_agent.run_travel_agent("What's the weather like?")
        assert first["missing"] == ["destination"]

        second = await navigo_agent.run_travel_agent("no idea", thread_id=first["thread_id"])
        assert second["needs_clarification"] is False
        assert second["answer"] == "FINAL PLAN"

    @pytest.mark.asyncio
    async def test_finished_thread_is_not_reused(self, graph_env):
        graph_env["extract"].return_value = ("weather", TripSlots(destination="Tokyo"))
        first = await navigo_agent.run_travel_agent("Weather in Tokyo?")
        second = await navigo_agent.run_travel_agent("Weather in Tokyo?", thread_id=first["thread_id"])
        assert second["thread_id"] != first["thread_id"]
        assert second["llm_calls"] == first["llm_calls"]  # no carry-over between runs

    @pytest.mark.asyncio
    async def test_unknown_thread_id_starts_new_run(self, graph_env):
        graph_env["extract"].return_value = ("weather", TripSlots(destination="Tokyo"))
        result = await navigo_agent.run_travel_agent("Weather in Tokyo?", thread_id="user_does_not_exist")
        assert result["thread_id"] != "user_does_not_exist"
        assert result["answer"] == "FINAL PLAN"

    @pytest.mark.asyncio
    async def test_allow_clarification_false_never_pauses(self, graph_env):
        graph_env["extract"].return_value = ("full_trip", TripSlots())
        result = await navigo_agent.run_travel_agent("Plan something", allow_clarification=False)
        assert result["needs_clarification"] is False
        assert result["answer"] == "FINAL PLAN"

    @pytest.mark.asyncio
    async def test_general_intent_never_pauses(self, graph_env):
        graph_env["extract"].return_value = ("general", TripSlots())
        result = await navigo_agent.run_travel_agent("Hello")
        assert result["needs_clarification"] is False


class TestClarificationStreaming:
    @pytest.mark.asyncio
    async def test_stream_emits_clarification_then_completes_on_resume(self, graph_env):
        graph_env["extract"].side_effect = [
            ("flight", TripSlots(destination="Dubai")),
            ("flight", TripSlots(origin="Dhaka")),
        ]

        first = _parse_sse([c async for c in streaming.stream_travel_plan("Flight info for Dubai")])
        events = [e for e, _ in first]
        assert events[-1] == "clarification"
        assert "complete" not in events
        payload = first[-1][1]
        assert payload["missing"] == ["origin"]
        assert "departing from" in payload["question"]
        thread_id = payload["thread_id"]

        second = _parse_sse([c async for c in streaming.stream_travel_plan("From Dhaka", thread_id)])
        events = [e for e, _ in second]
        assert "clarification" not in events
        assert events[-1] == "complete"
        complete = second[-1][1]
        assert complete["thread_id"] == thread_id
        assert complete["answer"] == "FINAL PLAN"


class TestClarificationApi:
    def test_api_returns_question(self):
        from fastapi.testclient import TestClient

        from app import app

        paused = {
            "thread_id": "user_abc", "answer": "Where would you like to go?",
            "needs_clarification": True, "question": "Where would you like to go?",
            "missing": ["destination"], "llm_calls": 1,
        }
        with patch("app.run_travel_agent", AsyncMock(return_value=paused)) as run:
            response = TestClient(app).post("/api/travel", json={"message": "weather?", "thread_id": None})
            data = response.json()
            assert response.status_code == 200
            assert data["success"] is True
            assert data["needs_clarification"] is True
            assert data["question"] == "Where would you like to go?"
            assert data["thread_id"] == "user_abc"

            # The reply is sent back with the same thread_id
            TestClient(app).post("/api/travel", json={"message": "Tokyo", "thread_id": "user_abc"})
            assert run.await_args.kwargs == {"user_input": "Tokyo", "thread_id": "user_abc"}


class TestStreamProgressEvents:
    @pytest.mark.asyncio
    async def test_every_node_reports_started_then_completed(self, graph_env):
        graph_env["extract"].return_value = ("full_trip", TripSlots(destination="Tokyo", duration_days=5))
        parsed = _parse_sse([c async for c in streaming.stream_travel_plan("Plan a 5-day trip to Tokyo")])
        progress = [(d["agent"], d["status"]) for e, d in parsed if e == "progress"]

        # Each node: equal numbers of started/completed, never completed before started
        for agent in {a for a, _ in progress}:
            statuses = [s for a, s in progress if a == agent]
            assert statuses.count("started") == statuses.count("completed"), agent
            assert statuses[0] == "started", agent
            assert all(a != b for a, b in itertools.pairwise(statuses)), agent

        # The supervisor runs twice (before and after the parallel batch)
        assert [s for a, s in progress if a == "supervisor"] == ["started", "completed"] * 2
        # The three data agents are all running before any of them finishes
        data_agents = {"flight_agent", "hotel_agent", "weather_agent"}
        data_events = [(a, s) for a, s in progress if a in data_agents]
        assert {a for a, s in data_events[:3]} == data_agents
        assert all(s == "started" for _, s in data_events[:3])
        assert parsed[-1][0] == "complete"
