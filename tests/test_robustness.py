"""Robustness: orchestrator plan guard, timeouts, forecast coverage, run summary log."""

import asyncio
import logging
import time
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import navigo_agent
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.orchestrator as orchestrator_mod
from navigo_agent.graph.agents import react_utils
from navigo_agent.graph.agents.weather import forecast_coverage_note
from navigo_agent.graph.orchestrator import (
    APOLOGY_ANSWER,
    default_plan,
    orchestrator_node,
)
from navigo_agent.graph.slots import TripSlots
from navigo_agent.session import log_run_summary
from navigo_agent.state import build_initial_state
from tests.fakes import tool_call


def _state(intent, **extra):
    state = build_initial_state("q")
    state["user_intent"] = intent
    state.update(extra)
    return state


def _llm(*results):
    """An orchestrator LLM whose calls return (or raise) `results` in order; the last one repeats."""
    queue = list(results)

    async def _ainvoke(_messages):
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, AIMessage) else AIMessage(content=result)

    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.ainvoke = AsyncMock(side_effect=_ainvoke)
    return llm


def _delegated(update) -> list[str]:
    return [call["name"] for call in update["messages"][-1].tool_calls]


class TestDefaultPlan:
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
        assert default_plan(_state(intent)) == expected

    def test_completed_agents_are_excluded(self):
        state = _state("full_trip", completed_agents=["flight_agent", "weather_agent"])
        assert default_plan(state) == ["hotel_agent"]


class TestOrchestratorPlanGuard:
    async def _run(self, state, *llm_results):
        llm = _llm(*llm_results)
        with patch.object(orchestrator_mod, "get_llm", lambda **_: llm):
            return await orchestrator_node(state), llm

    @pytest.mark.asyncio
    async def test_llm_error_uses_default_plan(self):
        update, _ = await self._run(_state("weather"), RuntimeError("rate limited"))
        assert _delegated(update) == ["weather_agent"]
        assert update["errors"] == ["orchestrator_llm_error: rate limited"]
        assert update["llm_calls"] == 0  # the failed call is not counted
        assert update["orchestrator_rounds"] == 1

    @pytest.mark.asyncio
    async def test_unusable_output_uses_default_plan(self):
        """Neither a valid delegation nor an answer: empty text, or only unknown tools."""
        for response in ("", AIMessage(content="", tool_calls=[tool_call("book_flight")])):
            update, _ = await self._run(_state("full_trip"), response)
            assert _delegated(update) == ["flight_agent", "hotel_agent", "weather_agent"]
            assert update["llm_calls"] == 1

    @pytest.mark.asyncio
    async def test_answering_with_no_data_is_overridden(self):
        update, _ = await self._run(_state("hotel"), "Here are some hotels I made up.")
        assert _delegated(update) == ["hotel_agent"]
        assert update["messages"][-1].content == ""  # the premature answer is dropped
        assert update["errors"] == []

    @pytest.mark.asyncio
    async def test_answering_after_data_is_respected(self):
        state = _state("weather", weather_results="sunny", completed_agents=["weather_agent"])
        update, llm = await self._run(state, "It is sunny in Tokyo.")
        assert update["messages"][-1].content == "It is sunny in Tokyo."
        assert not update["messages"][-1].tool_calls
        assert "orchestrator_rounds" not in update  # no new delegation round
        assert llm.ainvoke.await_count == 1  # the decision call's text IS the answer

    @pytest.mark.asyncio
    async def test_general_intent_may_answer_with_no_data(self):
        update, _ = await self._run(_state("general"), "Hello! Where would you like to go?")
        assert update["messages"][-1].content == "Hello! Where would you like to go?"

    @pytest.mark.asyncio
    async def test_failed_decision_falls_through_to_a_tool_free_answer(self):
        """After the only planned agent ran, a failed decision call has nothing left to delegate."""
        state = _state("weather", completed_agents=["weather_agent"], weather_results="unavailable")
        update, llm = await self._run(state, RuntimeError("tool_use_failed"), "No weather data, sorry.")
        assert update["messages"][-1].content == "No weather data, sorry."
        assert update["errors"] == ["orchestrator_llm_error: tool_use_failed"]
        assert update["llm_calls"] == 1
        assert llm.bind_tools.call_args.kwargs == {"tool_choice": "none"}

    @pytest.mark.asyncio
    async def test_total_outage_ends_with_an_apology_not_a_crash(self):
        state = _state("weather", completed_agents=["weather_agent"], weather_results="unavailable")
        update, _ = await self._run(state, RuntimeError("down"))
        assert update["messages"][-1].content == APOLOGY_ANSWER
        assert update["errors"] == ["orchestrator_llm_error: down", "orchestrator_final_answer: down"]
        assert update["llm_calls"] == 0

    @pytest.mark.asyncio
    async def test_round_budget_stops_delegation(self, monkeypatch):
        monkeypatch.setattr(orchestrator_mod, "MAX_ORCHESTRATOR_ROUNDS", 1)
        state = _state("weather", weather_results="sunny", completed_agents=["weather_agent"],
                       orchestrator_rounds=1)
        update, llm = await self._run(state, AIMessage(content="", tool_calls=[tool_call("hotel_agent")]), "done")
        # The only call made has tools switched off — the model cannot delegate again.
        assert llm.bind_tools.call_args_list[-1].kwargs == {"tool_choice": "none"}
        assert not update["messages"][-1].tool_calls

    @pytest.mark.asyncio
    async def test_broken_tool_calling_still_produces_a_plan_end_to_end(self, graph_env):
        """Every tool-calling request fails; plain completions work. The plan guard gathers
        the data and the orchestrator still writes the answer."""

        class _NoToolCalling:
            def bind_tools(self, _tools, tool_choice=None):
                bound = AsyncMock()
                if tool_choice == "none":
                    bound.ainvoke.return_value = AIMessage(content="PLAN FROM REPORTS")
                else:
                    bound.ainvoke.side_effect = RuntimeError("tool_use_failed")
                return bound

        graph_env["extract"].return_value = ("full_trip", TripSlots(destination="Tokyo", duration_days=5))
        with patch.object(orchestrator_mod, "get_llm", lambda **_: _NoToolCalling()):
            result = await navigo_agent.run_travel_agent("Plan a 5-day trip to Tokyo")
        assert result["flight_results"] == "FLIGHTS"
        assert result["hotel_results"] == "HOTELS"
        assert "sunny" in result["weather_results"]
        assert result["itinerary"] == "ITINERARY"
        assert result["answer"] == "PLAN FROM REPORTS"
        assert result["errors"] == ["orchestrator_llm_error: tool_use_failed"] * 2


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
        # (the fixture's orchestrator always delegates to all three research agents)
        assert "weather_agent" in summaries[0].agents
        assert summaries[0].llm_calls == result["llm_calls"]
