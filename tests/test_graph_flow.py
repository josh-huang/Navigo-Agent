"""Graph-level tests: routing through the real StateGraph with mocked LLMs/tools.

No network, no PostgreSQL — the graph is compiled with an in-memory checkpointer.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import navigo_agent.graph.agents.final as final_mod
import navigo_agent.graph.agents.flight as flight_mod
import navigo_agent.graph.agents.hotel as hotel_mod
import navigo_agent.graph.agents.itinerary as itinerary_mod
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.supervisor as supervisor_mod
from navigo_agent.graph.builder import build_travel_graph
from navigo_agent.graph.routing import route_after_supervisor

# Low limit so a routing loop fails fast instead of spinning.
RECURSION_LIMIT = 30


def _initial_state(query: str, destination: str | None = "Tokyo") -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "user_intent": "",
        "extracted_destination": destination,
        "supervisor_reasoning": "",
        "supervisor_loop_count": 0,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "pending_agents": [],
        "completed_agents": [],
        "llm_calls": 0,
        "errors": [],
    }


class ScriptedSupervisorLLM:
    """Dispatches `first_dispatch` on the first call, then says 'finalize'."""

    def __init__(self, first_dispatch: list[str]):
        self.first_dispatch = first_dispatch
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        agents = self.first_dispatch if self.calls == 1 else []
        return AIMessage(content=json.dumps({
            "reasoning": "scripted",
            "next_action": "dispatch" if agents else "synthesize_final",
            "agents_to_dispatch": agents,
            "is_complete": not agents,
        }))


class FinalLLM:
    async def ainvoke(self, _messages):
        return AIMessage(content="FINAL PLAN")


def _fake_react(answer: str, errors: list[str] | None = None):
    async def _run(**kwargs):
        return answer, kwargs.get("llm_calls", 0) + 1, list(errors or [])

    return _run


async def _run_graph(query, supervisor_llm, *, weather=None, forecast=None, react_errors=None):
    """Run the full graph and return (visited node names, final state)."""
    weather = weather or AsyncMock(return_value="sunny, 22C")
    forecast = forecast or AsyncMock(return_value="mild all week")

    with (
        patch.object(supervisor_mod, "get_llm", lambda **_: supervisor_llm),
        patch.object(final_mod, "get_llm", lambda **_: FinalLLM()),
        patch.object(flight_mod, "get_llm", lambda **_: None),
        patch.object(hotel_mod, "get_llm", lambda **_: None),
        patch.object(itinerary_mod, "get_llm", lambda **_: None),
        patch.object(flight_mod, "run_react_loop", _fake_react("FLIGHTS", react_errors)),
        patch.object(hotel_mod, "run_react_loop", _fake_react("HOTELS", react_errors)),
        patch.object(itinerary_mod, "run_react_loop", _fake_react("ITINERARY")),
        patch.object(weather_mod, "weather_mcp_search", weather),
        patch.object(weather_mod, "forecast_mcp_search", forecast),
    ):
        graph = build_travel_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test"}, "recursion_limit": RECURSION_LIMIT}
        visited: list[str] = []
        async for update in graph.astream(_initial_state(query), config=config, stream_mode="updates"):
            visited.extend(update.keys())
        state = (await graph.aget_state(config)).values
    return visited, state


class TestSingleIntentRouting:
    @pytest.mark.asyncio
    async def test_weather_only_does_not_run_itinerary(self):
        visited, state = await _run_graph(
            "What is the weather in Tokyo right now?",
            ScriptedSupervisorLLM(["weather_agent"]),
        )
        assert "weather_agent" in visited
        assert "itinerary_agent" not in visited
        assert visited[-1] == "final_synthesizer"
        assert state["itinerary"] == ""
        assert state["messages"][-1].content == "FINAL PLAN"

    @pytest.mark.asyncio
    async def test_flight_only_does_not_run_itinerary(self):
        visited, _ = await _run_graph(
            "Find me flights from Dhaka to Dubai",
            ScriptedSupervisorLLM(["flight_agent"]),
        )
        assert visited.count("flight_agent") == 1
        assert "itinerary_agent" not in visited

    @pytest.mark.asyncio
    async def test_hotel_only_does_not_run_itinerary(self):
        visited, _ = await _run_graph(
            "Find budget hotels in Bangkok",
            ScriptedSupervisorLLM(["hotel_agent"]),
        )
        assert visited.count("hotel_agent") == 1
        assert "itinerary_agent" not in visited


class TestFullTripRouting:
    @pytest.mark.asyncio
    async def test_full_trip_runs_all_agents_then_itinerary(self):
        visited, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedSupervisorLLM(["flight_agent", "hotel_agent", "weather_agent"]),
        )
        for agent in ("flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"):
            assert visited.count(agent) == 1, f"{agent} ran {visited.count(agent)} times"
        assert visited.index("itinerary_agent") > visited.index("weather_agent")
        assert visited[-1] == "final_synthesizer"
        assert state["itinerary"] == "ITINERARY"
        assert state["errors"] == []

    @pytest.mark.asyncio
    async def test_errors_are_not_duplicated(self):
        """`errors` uses an operator.add reducer — agents must return only new ones."""
        _, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedSupervisorLLM(["flight_agent", "hotel_agent", "weather_agent"]),
            react_errors=["search failed"],
        )
        # One from flight_agent, one from hotel_agent — no doubling.
        assert state["errors"] == ["search failed", "search failed"]


class TestWeatherFailure:
    @pytest.mark.asyncio
    async def test_weather_tool_error_terminates(self):
        """A failing weather tool used to loop until GraphRecursionError."""
        visited, state = await _run_graph(
            "What is the weather in Atlantis?",
            ScriptedSupervisorLLM(["weather_agent"]),
            weather=AsyncMock(side_effect=RuntimeError("city not found")),
        )
        assert visited.count("weather_agent") == 1
        assert visited[-1] == "final_synthesizer"
        assert "unavailable" in state["weather_results"].lower()
        assert state["pending_agents"] == []
        assert state["errors"] == ["weather_agent: city not found"]

    @pytest.mark.asyncio
    async def test_weather_agent_leaves_pending_on_error(self):
        state = _initial_state("weather in Atlantis")
        state["pending_agents"] = ["weather_agent", "hotel_agent"]
        with patch.object(weather_mod, "weather_mcp_search", AsyncMock(side_effect=RuntimeError("boom"))):
            update = await weather_mod.weather_agent(state)
        assert update["pending_agents"] == ["hotel_agent"]
        assert "weather_agent" in update["completed_agents"]

    @pytest.mark.asyncio
    async def test_destination_extraction_error_is_handled(self):
        state = _initial_state("weather?", destination=None)
        state["pending_agents"] = ["weather_agent"]
        with patch.object(weather_mod, "extract_destination", side_effect=ValueError("no key")):
            update = await weather_mod.weather_agent(state)
        assert update["pending_agents"] == []
        assert update["errors"] == ["weather_agent: no key"]


class TestAgentErrorReporting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("module,node", [
        (flight_mod, "flight_agent"),
        (hotel_mod, "hotel_agent"),
        (itinerary_mod, "itinerary_agent"),
    ])
    async def test_agent_returns_only_new_errors(self, module, node):
        state = _initial_state("Plan a trip to Tokyo")
        state["errors"] = ["older error"]
        with (
            patch.object(module, "get_llm", lambda **_: None),
            patch.object(module, "run_react_loop", _fake_react("X", ["new error"])),
        ):
            update = await getattr(module, node)(state)
        assert update["errors"] == ["new error"]


class TestRouteAfterSupervisor:
    def _state(self, intent, **extra):
        return {"user_intent": intent, "pending_agents": [], "completed_agents": [], **extra}

    @pytest.mark.parametrize("intent", ["weather", "flights", "flight", "hotel", "general"])
    def test_single_topic_with_data_goes_to_final(self, intent):
        state = self._state(intent, weather_results="sunny")
        assert route_after_supervisor(state) == "final_synthesizer"

    @pytest.mark.parametrize("intent", ["full_trip", "itinerary"])
    def test_trip_intent_with_data_gets_itinerary(self, intent):
        state = self._state(intent, flight_results="F")
        assert route_after_supervisor(state) == "itinerary_agent"

    def test_trip_intent_with_itinerary_goes_to_final(self):
        state = self._state("full_trip", flight_results="F", itinerary="I", completed_agents=["itinerary_agent"])
        assert route_after_supervisor(state) == "final_synthesizer"

    def test_explicit_dispatch_is_honoured_for_any_intent(self):
        state = self._state("weather", weather_results="sunny")
        state["pending_agents"] = ["itinerary_agent"]
        assert route_after_supervisor(state) == "itinerary_agent"
