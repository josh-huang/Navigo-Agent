"""Graph-level tests: routing through the real StateGraph with mocked LLMs/tools.

No network, no PostgreSQL — the graph is compiled with an in-memory checkpointer.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import navigo_agent.graph.agents.final as final_mod
import navigo_agent.graph.agents.flight as flight_mod
import navigo_agent.graph.agents.hotel as hotel_mod
import navigo_agent.graph.agents.itinerary as itinerary_mod
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.routing as routing_mod
import navigo_agent.graph.supervisor as supervisor_mod
from navigo_agent.graph.builder import build_travel_graph
from navigo_agent.graph.routing import route_after_supervisor
from navigo_agent.graph.slots import TripSlots
from navigo_agent.state import build_initial_state

# Low limit so a routing loop fails fast instead of spinning.
RECURSION_LIMIT = 30


def _initial_state(query: str, destination: str | None = "Tokyo") -> dict:
    state = build_initial_state(query)
    state["trip"] = {"destination": destination}
    state["extracted_destination"] = destination
    return state


class ScriptedSupervisorLLM:
    """Dispatches one scripted batch per call, then says 'finalize'."""

    def __init__(self, *batches: list[str]):
        self.batches = list(batches)
        self.calls = 0

    async def ainvoke(self, _messages):
        agents = self.batches[self.calls] if self.calls < len(self.batches) else []
        self.calls += 1
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
    async def _run(**_kwargs):
        return answer, 1, list(errors or [])

    return _run


async def _run_graph(
    query, supervisor_llm, *, weather=None, forecast=None, react_errors=None, react=None, slots=None,
):
    """Run the full graph and return (visited node names, final state).

    `react` optionally overrides the fake ReAct loop per agent name.
    """
    react = react or {}
    slots = slots or TripSlots(destination="Tokyo")
    weather = weather or AsyncMock(return_value="sunny, 22C")
    forecast = forecast or AsyncMock(return_value="mild all week")

    with (
        patch.object(routing_mod, "extract_trip_request", AsyncMock(return_value=(None, slots))),
        patch.object(supervisor_mod, "get_llm", lambda **_: supervisor_llm),
        patch.object(final_mod, "get_llm", lambda **_: FinalLLM()),
        patch.object(flight_mod, "get_llm", lambda **_: None),
        patch.object(hotel_mod, "get_llm", lambda **_: None),
        patch.object(itinerary_mod, "get_llm", lambda **_: None),
        patch.object(flight_mod, "run_react_loop", react.get("flight_agent") or _fake_react("FLIGHTS", react_errors)),
        patch.object(hotel_mod, "run_react_loop", react.get("hotel_agent") or _fake_react("HOTELS", react_errors)),
        patch.object(itinerary_mod, "run_react_loop", _fake_react("ITINERARY")),
        patch.object(weather_mod, "weather_mcp_search", weather),
        patch.object(weather_mod, "forecast_mcp_search", forecast),
    ):
        graph = build_travel_graph().compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test"}, "recursion_limit": RECURSION_LIMIT}
        visited: list[str] = []
        async for update in graph.astream(build_initial_state(query), config=config, stream_mode="updates"):
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
    async def test_weather_agent_reports_completed_on_error(self):
        state = _initial_state("weather in Atlantis")
        with (
            patch.object(weather_mod, "weather_mcp_search", AsyncMock(side_effect=RuntimeError("boom"))),
            patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="fc")),
        ):
            update = await weather_mod.weather_agent(state)
        assert update["completed_agents"] == ["weather_agent"]
        # pending_agents belongs to the supervisor — parallel agents must not write it
        assert "pending_agents" not in update

    @pytest.mark.asyncio
    async def test_destination_extraction_error_is_handled(self):
        state = _initial_state("weather?", destination=None)
        with patch.object(weather_mod, "extract_destination", side_effect=ValueError("no key")):
            update = await weather_mod.weather_agent(state)
        assert update["completed_agents"] == ["weather_agent"]
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
        assert update["completed_agents"] == [node]
        assert update["llm_calls"] == 1  # delta, not a running total


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

    def test_multiple_data_agents_fan_out_as_list(self):
        state = self._state("full_trip")
        state["pending_agents"] = ["flight", "hotel_agent", "weather_agent", "hotel_agent", "bogus"]
        assert route_after_supervisor(state) == ["flight_agent", "hotel_agent", "weather_agent"]

    def test_itinerary_is_deferred_behind_data_agents(self):
        state = self._state("full_trip")
        state["pending_agents"] = ["itinerary_agent", "flight_agent"]
        assert route_after_supervisor(state) == ["flight_agent"]

    def test_explicit_dispatch_is_honoured_for_any_intent(self):
        state = self._state("weather", weather_results="sunny")
        state["pending_agents"] = ["itinerary_agent"]
        assert route_after_supervisor(state) == "itinerary_agent"


class TestParallelDispatch:
    @pytest.mark.asyncio
    async def test_data_agents_run_concurrently(self):
        """Each gated agent blocks until the other one has started too.

        Sequential execution would never release the gate → timeout.
        """
        started: list[str] = []
        all_started = asyncio.Event()

        def gated(name: str):
            async def _run(**_kwargs):
                started.append(name)
                if len(started) == 2:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=2)
                return name.upper(), 1, []

            return _run

        supervisor = ScriptedSupervisorLLM(["flight_agent", "hotel_agent", "weather_agent"])
        visited, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            supervisor,
            react={"flight_agent": gated("flight"), "hotel_agent": gated("hotel")},
        )

        assert sorted(started) == ["flight", "hotel"]
        assert state["flight_results"] == "FLIGHT"
        assert state["hotel_results"] == "HOTEL"
        # One fan-in: the supervisor runs once before the batch and once after it.
        assert visited.count("supervisor") == 2
        assert supervisor.calls == 2
        assert visited == [
            "intent_classifier", "supervisor",
            *visited[2:5],
            "supervisor", "itinerary_agent", "final_synthesizer",
        ]
        assert sorted(visited[2:5]) == ["flight_agent", "hotel_agent", "weather_agent"]

    @pytest.mark.asyncio
    async def test_parallel_writes_are_merged(self):
        _, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedSupervisorLLM(["flight_agent", "hotel_agent", "weather_agent"]),
        )
        assert sorted(state["completed_agents"]) == [
            "flight_agent", "hotel_agent", "itinerary_agent", "weather_agent",
        ]
        # intent(1) + supervisor(2) + flight(1) + hotel(1) + itinerary(1) + final(1)
        assert state["llm_calls"] == 7

    @pytest.mark.asyncio
    async def test_completed_agent_is_not_redispatched(self):
        supervisor = ScriptedSupervisorLLM(["weather_agent"], ["weather_agent"], ["weather_agent"])
        visited, _ = await _run_graph("What is the weather in Tokyo?", supervisor)
        assert visited.count("weather_agent") == 1
        assert visited[-1] == "final_synthesizer"

    @pytest.mark.asyncio
    async def test_one_agent_per_turn_still_works(self):
        """A supervisor that dispatches sequentially stays within the loop bound."""
        supervisor = ScriptedSupervisorLLM(["flight_agent"], ["hotel_agent"], ["weather_agent"])
        visited, _ = await _run_graph("Plan a 5-day trip to Tokyo", supervisor)
        for agent in ("flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"):
            assert visited.count(agent) == 1
        assert visited[-1] == "final_synthesizer"


class TestSlotsReachAgents:
    @pytest.mark.asyncio
    async def test_slots_are_stored_and_used(self):
        slots = TripSlots(
            origin="Dhaka", destination="Japan", destination_city="Tokyo",
            duration_days=7, budget="mid-range",
        )
        weather = AsyncMock(return_value="sunny")
        captured: dict = {}

        async def capture_flight(**kwargs):
            captured["flight_system"] = kwargs["conversation"][0].content
            return "FLIGHTS", 1, []

        _, state = await _run_graph(
            "Plan a 7 days Japan trip from Bangladesh",
            ScriptedSupervisorLLM(["flight_agent", "hotel_agent", "weather_agent"]),
            weather=weather, slots=slots, react={"flight_agent": capture_flight},
        )
        assert state["trip"]["origin"] == "Dhaka"
        assert state["extracted_destination"] == "Japan"
        # Weather uses the resolved city, not the country
        weather.assert_awaited_once_with("Tokyo")
        # Flight agent searches the route and sees the slots
        assert "Dhaka to Japan flights" in captured["flight_system"]
        assert "- Budget: mid-range" in captured["flight_system"]
        assert "- Duration (days): 7" in captured["flight_system"]
