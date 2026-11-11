"""Robustness: supervisor fallback plan, timeouts, forecast coverage, run summary log."""

import asyncio
import logging
import time
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import navigo_agent
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.supervisor as supervisor_mod
from navigo_agent.graph.agents import react_utils
from navigo_agent.graph.agents.weather import forecast_coverage_note
from navigo_agent.graph.slots import TripSlots
from navigo_agent.graph.supervisor import fallback_agents, supervisor_node
from navigo_agent.session import log_run_summary
from navigo_agent.state import build_initial_state


def _state(intent, **extra):
    state = build_initial_state("q")
    state["user_intent"] = intent
    state.update(extra)
    return state


def _llm(result):
    llm = AsyncMock()
    if isinstance(result, Exception):
        llm.ainvoke.side_effect = result
    else:
        llm.ainvoke.return_value = AIMessage(content=result)
    return llm


class TestFallbackPlan:
    @pytest.mark.parametrize(
        "intent,expected",
        [
            ("weather", ["weather_agent"]),
            ("flight", ["flight_agent"]),
            ("hotel", ["hotel_agent"]),
            ("full_trip", ["flight_agent", "hotel_agent", "weather_agent"]),
            ("itinerary", ["flight_agent", "hotel_agent", "weather_agent"]),
            ("general", []),
            ("something_new", ["flight_agent", "hotel_agent", "weather_agent"]),
        ],
    )
    def test_plan_by_intent(self, intent, expected):
        assert fallback_agents(_state(intent)) == expected

    def test_completed_agents_are_excluded(self):
        state = _state("full_trip", completed_agents=["flight_agent", "weather_agent"])
        assert fallback_agents(state) == ["hotel_agent"]


class TestSupervisorFallback:
    async def _run(self, llm_result, state):
        with patch.object(supervisor_mod, "get_llm", lambda **_: _llm(llm_result)):
            return await supervisor_node(state)

    @pytest.mark.asyncio
    async def test_llm_error_uses_default_plan(self):
        update = await self._run(RuntimeError("rate limited"), _state("weather"))
        assert update["pending_agents"] == ["weather_agent"]
        assert update["errors"] == ["supervisor_llm_error: rate limited"]
        assert "llm_calls" not in update  # the failed call is not counted

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", ["I think we should search flights.", "{broken json", ""])
    async def test_unparseable_output_uses_default_plan(self, raw):
        update = await self._run(raw, _state("full_trip"))
        assert update["pending_agents"] == ["flight_agent", "hotel_agent", "weather_agent"]
        assert update["errors"] == ["supervisor_parse_error"]
        assert update["llm_calls"] == 1

    @pytest.mark.asyncio
    async def test_finalizing_with_no_data_is_overridden(self):
        raw = '{"reasoning": "done", "next_action": "synthesize_final", "agents_to_dispatch": [], "is_complete": true}'
        update = await self._run(raw, _state("hotel"))
        assert update["pending_agents"] == ["hotel_agent"]
        assert "default plan" in update["supervisor_reasoning"]
        assert "errors" not in update

    @pytest.mark.asyncio
    async def test_finalizing_after_data_is_respected(self):
        raw = '{"reasoning": "done", "next_action": "synthesize_final", "agents_to_dispatch": [], "is_complete": true}'
        state = _state("weather", weather_results="sunny", completed_agents=["weather_agent"])
        update = await self._run(raw, state)
        assert update["pending_agents"] == []

    @pytest.mark.asyncio
    async def test_general_intent_may_finalize_with_no_data(self):
        raw = '{"reasoning": "chit-chat", "next_action": "synthesize_final", "agents_to_dispatch": [], "is_complete": true}'
        update = await self._run(raw, _state("general"))
        assert update["pending_agents"] == []

    @pytest.mark.asyncio
    async def test_failed_agent_is_not_retried_by_fallback(self):
        """After the only planned agent ran (even if it failed), fallback has nothing left."""
        state = _state("weather", completed_agents=["weather_agent"], weather_results="")
        update = await self._run(RuntimeError("down"), state)
        assert update["pending_agents"] == []

    @pytest.mark.asyncio
    async def test_supervisor_outage_still_produces_a_plan_end_to_end(self, graph_env):
        """With the supervisor LLM down for the whole run, the graph still gathers data."""
        graph_env["extract"].return_value = ("full_trip", TripSlots(destination="Tokyo", duration_days=5))
        with patch.object(supervisor_mod, "get_llm", lambda **_: _llm(RuntimeError("supervisor down"))):
            result = await navigo_agent.run_travel_agent("Plan a 5-day trip to Tokyo")
        assert result["flight_results"] == "FLIGHTS"
        assert result["hotel_results"] == "HOTELS"
        assert "sunny" in result["weather_results"]
        assert result["itinerary"] == "ITINERARY"
        assert result["answer"] == "FINAL PLAN"


class _OneSearchLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools, tool_choice=None):
        return self

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "web_search", "args": {"query": "q"}, "id": "1", "type": "tool_call"},
            ])
        return AIMessage(content="answer from what I know")


class TestTimeouts:
    @pytest.mark.asyncio
    async def test_hung_search_times_out(self, monkeypatch):
        monkeypatch.setattr(react_utils, "SEARCH_TIMEOUT_SECONDS", 0.05)

        async def hung(_query):
            await asyncio.sleep(10)

        conversation = [SystemMessage(content="s"), HumanMessage(content="h")]
        started = time.perf_counter()
        answer, _, errors = await react_utils.run_react_loop(
            llm=_OneSearchLLM(), conversation=conversation, search_fn=hung, agent_name="flight_agent",
        )
        assert time.perf_counter() - started < 2
        assert answer == "answer from what I know"
        assert errors == ["flight_agent(search): timed out after 0.05s"]
        assert "no response within" in conversation[-1].content

    @pytest.mark.asyncio
    async def test_hung_weather_tool_times_out(self, monkeypatch):
        monkeypatch.setattr(weather_mod, "WEATHER_TIMEOUT_SECONDS", 0.05)

        async def hung(_city):
            await asyncio.sleep(10)

        state = build_initial_state("weather in Tokyo")
        state["trip"] = {"destination": "Tokyo"}
        with (
            patch.object(weather_mod, "weather_mcp_search", hung),
            patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="fc")),
        ):
            update = await weather_mod.weather_agent(state)
        assert "unavailable" in update["weather_results"].lower()
        assert update["errors"] == ["weather_agent: no response within 0.05s"]
        assert update["completed_agents"] == ["weather_agent"]


class TestForecastCoverage:
    TODAY = date(2026, 11, 11)

    @pytest.mark.parametrize("start", [None, "", "not-a-date", "2026-11-11", "2026-11-16", "2026-01-01"])
    def test_no_note_when_dates_unknown_or_within_window(self, start):
        assert forecast_coverage_note(start, today=self.TODAY) == ""

    def test_note_when_trip_is_beyond_the_forecast(self):
        note = forecast_coverage_note("2027-03-01", today=self.TODAY)
        assert "2027-03-01" in note
        assert "110 days from now" in note
        assert "NOT a forecast for the travel dates" in note
        assert "seasonal climate" in note

    @pytest.mark.asyncio
    async def test_weather_agent_appends_the_note(self):
        state = build_initial_state("Plan a trip to Tokyo next spring")
        state["trip"] = {"destination": "Tokyo", "start_date": "2999-04-01"}
        with (
            patch.object(weather_mod, "weather_mcp_search", AsyncMock(return_value="sunny")),
            patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="mild")),
        ):
            update = await weather_mod.weather_agent(state)
        assert "sunny" in update["weather_results"]
        assert "NOT a forecast for the travel dates" in update["weather_results"]


class TestRunSummaryLog:
    def test_summary_fields(self, caplog):
        state = {"user_intent": "weather", "completed_agents": ["weather_agent"], "llm_calls": 3,
                 "errors": ["weather_agent: boom"]}
        with caplog.at_level(logging.INFO, logger="navigo_agent.session"):
            log_run_summary("user_abc", state, time.perf_counter() - 1.5, outcome="completed")
        record = caplog.records[-1]
        assert record.thread_id == "user_abc"
        assert record.outcome == "completed"
        assert record.intent == "weather"
        assert record.agents == ["weather_agent"]
        assert record.llm_calls == 3
        assert record.errors == ["weather_agent: boom"]
        assert 1400 <= record.duration_ms <= 5000
        assert "intent=weather" in record.getMessage()

    def test_tolerates_missing_state(self, caplog):
        with caplog.at_level(logging.INFO, logger="navigo_agent.session"):
            log_run_summary("user_abc", None, time.perf_counter(), outcome="failed")
        assert caplog.records[-1].outcome == "failed"
        assert caplog.records[-1].llm_calls == 0

    @pytest.mark.asyncio
    async def test_every_run_is_logged(self, graph_env, caplog):
        graph_env["extract"].return_value = ("weather", TripSlots(destination="Tokyo"))
        with caplog.at_level(logging.INFO, logger="navigo_agent.session"):
            result = await navigo_agent.run_travel_agent("Weather in Tokyo?")
        summaries = [r for r in caplog.records if getattr(r, "outcome", None)]
        assert len(summaries) == 1
        assert summaries[0].thread_id == result["thread_id"]
        assert summaries[0].outcome == "completed"
        # (the fixture's scripted supervisor always dispatches all three data agents)
        assert "weather_agent" in summaries[0].agents
        assert summaries[0].llm_calls == result["llm_calls"]
