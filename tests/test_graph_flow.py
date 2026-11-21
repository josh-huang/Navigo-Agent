"""Graph-level tests: the orchestrator loop through the real StateGraph with mocked LLMs/tools.

No network, no PostgreSQL — the graph is compiled with an in-memory checkpointer.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Send

import navigo_agent.graph.agents.flight as flight_mod
import navigo_agent.graph.agents.hotel as hotel_mod
import navigo_agent.graph.agents.itinerary as itinerary_mod
import navigo_agent.graph.agents.weather as weather_mod
import navigo_agent.graph.intake as intake_mod
import navigo_agent.graph.orchestrator as orchestrator_mod
from navigo_agent.graph.agents import SUBAGENT_NODES
from navigo_agent.graph.agents.delegation import SUBAGENT_NAMES, delegation_tools
from navigo_agent.graph.builder import build_travel_graph
from navigo_agent.graph.orchestrator import (
    _required_delegations,
    _select_delegations,
    pending_delegations,
    route_after_orchestrator,
)
from navigo_agent.graph.slots import TripSlots
from navigo_agent.state import build_initial_state
from tests.fakes import ScriptedOrchestratorLLM, tool_call

# Low limit so a delegation loop fails fast instead of spinning.
RECURSION_LIMIT = 30

RESEARCH = ["flight_agent", "hotel_agent", "weather_agent"]
INTAKE = ["intent_classifier", "prepare_clarification", "clarify"]


def _initial_state(query: str, destination: str | None = "Tokyo") -> dict:
    state = build_initial_state(query)
    state["trip"] = {"destination": destination}
    state["extracted_destination"] = destination
    return state


def _fake_react(answer: str, errors: list[str] | None = None):
    async def _run(**_kwargs):
        return answer, 1, list(errors or [])

    return _run


async def _run_graph(
    query, orchestrator_llm, *, weather=None, forecast=None, react_errors=None, react=None, slots=None,
    intent=None,
):
    """Run the full graph and return (visited node names, final state).

    `react` optionally overrides the fake ReAct loop per agent name; `intent` is what the
    extractor returns (None → the regex fast path / the full-trip default decides).
    """
    react = react or {}
    # Complete enough that the clarify node never pauses in these tests.
    slots = slots or TripSlots(origin="Dhaka", destination="Tokyo", duration_days=5)
    weather = weather or AsyncMock(return_value="sunny, 22C")
    forecast = forecast or AsyncMock(return_value="mild all week")

    with (
        patch.object(intake_mod, "extract_trip_request", AsyncMock(return_value=(intent, slots))),
        patch.object(orchestrator_mod, "get_llm", lambda **_: orchestrator_llm),
        patch.object(flight_mod, "get_llm", lambda **_: None),
        patch.object(hotel_mod, "get_llm", lambda **_: None),
        patch.object(itinerary_mod, "get_llm", lambda **_: None),
        patch.object(flight_mod, "run_react_loop", react.get("flight_agent") or _fake_react("FLIGHTS", react_errors)),
        patch.object(hotel_mod, "run_react_loop", react.get("hotel_agent") or _fake_react("HOTELS", react_errors)),
        patch.object(itinerary_mod, "run_react_loop", react.get("itinerary_agent") or _fake_react("ITINERARY")),
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


class TestGraphShape:
    def test_orchestrator_is_the_hub(self):
        """Every sub-agent is reachable only from the orchestrator and reports back to it."""
        edges = {(e.source, e.target) for e in build_travel_graph().compile().get_graph().edges}
        for agent in SUBAGENT_NAMES:
            assert ("orchestrator", agent) in edges
            assert (agent, "orchestrator") in edges
            assert {target for source, target in edges if source == agent} == {"orchestrator"}
        assert ("orchestrator", "__end__") in edges
        assert ("clarify", "orchestrator") in edges

    def test_registry_nodes_and_tools_agree(self):
        assert tuple(SUBAGENT_NODES) == SUBAGENT_NAMES
        tools = delegation_tools()
        assert [t["function"]["name"] for t in tools] == list(SUBAGENT_NAMES)
        assert [t["function"]["name"] for t in delegation_tools(["weather_agent"])] == ["weather_agent"]
        assert all(t["function"]["description"] and t["function"]["parameters"]["required"] for t in tools)


class TestSingleTopicRequests:
    @pytest.mark.asyncio
    async def test_weather_only_does_not_run_itinerary(self):
        visited, state = await _run_graph(
            "What is the weather in Tokyo right now?",
            ScriptedOrchestratorLLM(["weather_agent"]),
        )
        assert visited == [*INTAKE, "orchestrator", "weather_agent", "orchestrator"]
        assert state["itinerary"] == ""
        assert state["messages"][-1].content == "FINAL PLAN"
        assert state["orchestrator_rounds"] == 1

    @pytest.mark.asyncio
    async def test_flight_only_does_not_run_itinerary(self):
        visited, _ = await _run_graph(
            "Find me flights from Dhaka to Dubai",
            ScriptedOrchestratorLLM(["flight_agent"]),
        )
        assert visited.count("flight_agent") == 1
        assert "itinerary_agent" not in visited

    @pytest.mark.asyncio
    async def test_hotel_only_does_not_run_itinerary(self):
        visited, _ = await _run_graph(
            "Find budget hotels in Bangkok",
            ScriptedOrchestratorLLM(["hotel_agent"]),
        )
        assert visited.count("hotel_agent") == 1
        assert "itinerary_agent" not in visited

    @pytest.mark.asyncio
    async def test_general_question_is_answered_without_sub_agents(self):
        """The orchestrator may answer on its own — no forced delegation for `general`."""
        llm = ScriptedOrchestratorLLM(answer="Tokyo is the capital of Japan.")
        visited, state = await _run_graph("Tell me about Tokyo", llm, intent="general")
        assert visited == [*INTAKE, "orchestrator"]
        assert state["completed_agents"] == []
        assert state["messages"][-1].content == "Tokyo is the capital of Japan."
        assert state["llm_calls"] == 2  # intent + one orchestrator call


class TestFullTrip:
    @pytest.mark.asyncio
    async def test_research_then_itinerary_then_answer(self):
        visited, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"]),
        )
        for agent in SUBAGENT_NAMES:
            assert visited.count(agent) == 1, f"{agent} ran {visited.count(agent)} times"
        assert visited.index("itinerary_agent") > visited.index("weather_agent")
        assert visited[-1] == "orchestrator"
        assert state["itinerary"] == "ITINERARY"
        assert state["messages"][-1].content == "FINAL PLAN"
        assert state["errors"] == []

    @pytest.mark.asyncio
    async def test_itinerary_is_built_even_if_the_orchestrator_forgets_it(self):
        """Plan guard: a trip answer written without an itinerary is not accepted."""
        llm = ScriptedOrchestratorLLM(RESEARCH)  # answers right after the research round
        visited, state = await _run_graph("Plan a 5-day trip to Tokyo", llm)
        assert visited.count("itinerary_agent") == 1
        assert visited[-2:] == ["itinerary_agent", "orchestrator"]
        assert state["itinerary"] == "ITINERARY"

    @pytest.mark.asyncio
    async def test_itinerary_called_with_research_is_held_back_a_round(self):
        """The itinerary builds on the research reports, so it never shares their round."""
        seen: dict = {}

        async def capture(**kwargs):
            seen["system"] = kwargs["conversation"][0].content
            return "ITINERARY", 1, []

        llm = ScriptedOrchestratorLLM([*RESEARCH, "itinerary_agent"], ["itinerary_agent"])
        visited, _ = await _run_graph("Plan a 5-day trip to Tokyo", llm, react={"itinerary_agent": capture})
        assert visited == [*INTAKE, "orchestrator", *visited[4:7], "orchestrator", "itinerary_agent", "orchestrator"]
        assert sorted(visited[4:7]) == RESEARCH
        # ...and by then it sees its peers' reports
        assert "FLIGHTS" in seen["system"] and "HOTELS" in seen["system"] and "sunny, 22C" in seen["system"]

    @pytest.mark.asyncio
    async def test_errors_are_not_duplicated(self):
        """`errors` uses an operator.add reducer — agents must return only new ones."""
        _, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"]),
            react_errors=["search failed"],
        )
        # One from flight_agent, one from hotel_agent — no doubling.
        assert state["errors"] == ["search failed", "search failed"]


class TestDelegationTranscript:
    """The orchestrator talks to its sub-agents through tool calls and ToolMessages."""

    @pytest.mark.asyncio
    async def test_every_delegation_is_answered_by_a_report(self):
        _, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"]),
        )
        messages = state["messages"]
        assert isinstance(messages[0], HumanMessage)
        called = {c["id"]: c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
        reports = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        assert sorted(called.values()) == sorted(SUBAGENT_NAMES)
        assert set(reports) == set(called)
        assert all(reports[call_id].name == name for call_id, name in called.items())
        by_agent = {m.name: m.content for m in reports.values()}
        assert by_agent["flight_agent"] == "FLIGHTS" == state["flight_results"]
        assert by_agent["itinerary_agent"] == "ITINERARY" == state["itinerary"]
        assert by_agent["weather_agent"] == state["weather_results"]
        # The run ends on the orchestrator's own answer, not on a tool call
        assert isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls
        assert state["delegation"] == {}  # a Send payload, never written back

    @pytest.mark.asyncio
    async def test_orchestrator_reads_the_reports_on_its_next_turn(self):
        llm = ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"])
        await _run_graph("Plan a 5-day trip to Tokyo", llm)
        first, second, third = (messages for messages, _ in llm.calls)

        assert isinstance(first[0], SystemMessage) and "main agent" in first[0].content
        assert "Reports received from: none yet" in first[1].content
        assert len(first) == 2  # nothing delegated yet

        # Turn 2: its own tool calls, then one report per sub-agent
        assert [type(m) for m in second[2:]] == [AIMessage, ToolMessage, ToolMessage, ToolMessage]
        assert {m.content for m in second[3:]} >= {"FLIGHTS", "HOTELS"}
        assert "Reports received from: flight_agent, hotel_agent, weather_agent" in second[1].content
        assert "Sub-agents not used yet: itinerary_agent" in second[1].content

        # Turn 3: everything has reported — the answer is written with tools switched off
        assert third[-2].content == "ITINERARY"
        assert "Do not call any tools" in third[-1].content
        assert [choice for _, choice in llm.calls] == [None, None, "none"]

    @pytest.mark.asyncio
    async def test_brief_reaches_the_sub_agent(self):
        seen: dict = {}

        async def capture(**kwargs):
            seen["human"] = kwargs["conversation"][1].content
            return "FLIGHTS", 1, []

        await _run_graph(
            "Find me flights from Dhaka to Dubai",
            ScriptedOrchestratorLLM(["flight_agent"]),
            react={"flight_agent": capture},
        )
        assert "Find flight information for: Find me flights from Dhaka to Dubai" in seen["human"]
        assert "## Task From The Orchestrator" in seen["human"]
        assert "brief for flight_agent" in seen["human"]


class TestWeatherFailure:
    @pytest.mark.asyncio
    async def test_weather_tool_error_terminates(self):
        """A failing sub-agent still reports back, so the loop cannot spin."""
        visited, state = await _run_graph(
            "What is the weather in Atlantis?",
            ScriptedOrchestratorLLM(["weather_agent"]),
            weather=AsyncMock(side_effect=RuntimeError("city not found")),
        )
        assert visited.count("weather_agent") == 1
        assert visited[-1] == "orchestrator"
        assert "unavailable" in state["weather_results"].lower()
        assert pending_delegations(state) == []
        assert state["errors"] == ["weather_agent: city not found"]

    @pytest.mark.asyncio
    async def test_weather_agent_reports_back_on_error(self):
        state = _initial_state("weather in Atlantis")
        state["delegation"] = {"id": "call_1", "args": {"city": "Atlantis"}}
        with (
            patch.object(weather_mod, "weather_mcp_search", AsyncMock(side_effect=RuntimeError("boom"))),
            patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="fc")),
        ):
            update = await weather_mod.weather_agent(state)
        assert update["completed_agents"] == ["weather_agent"]
        (message,) = update["messages"]
        assert isinstance(message, ToolMessage)
        assert (message.tool_call_id, message.name) == ("call_1", "weather_agent")
        assert "unavailable" in message.content
        # Orchestration state belongs to the orchestrator — sub-agents must not write it
        assert "orchestrator_rounds" not in update
        assert "delegation" not in update

    @pytest.mark.asyncio
    async def test_destination_extraction_error_is_handled(self):
        state = _initial_state("weather?", destination=None)
        with patch.object(weather_mod, "extract_destination", side_effect=ValueError("no key")):
            update = await weather_mod.weather_agent(state)
        assert update["completed_agents"] == ["weather_agent"]
        assert update["errors"] == ["weather_agent: no key"]

    @pytest.mark.asyncio
    async def test_delegated_city_is_used_when_slots_have_none(self):
        state = _initial_state("what's it like there?", destination=None)
        state["delegation"] = {"id": "call_1", "args": {"city": "Lisbon"}}
        weather = AsyncMock(return_value="sunny")
        with (
            patch.object(weather_mod, "weather_mcp_search", weather),
            patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="fc")),
            patch.object(weather_mod, "extract_destination", side_effect=AssertionError("no LLM call expected")),
        ):
            update = await weather_mod.weather_agent(state)
        weather.assert_awaited_once_with("Lisbon")
        assert update["llm_calls"] == 0


class TestAgentErrorReporting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("module,node,result_key", [
        (flight_mod, "flight_agent", "flight_results"),
        (hotel_mod, "hotel_agent", "hotel_results"),
        (itinerary_mod, "itinerary_agent", "itinerary"),
    ])
    async def test_agent_returns_only_deltas(self, module, node, result_key):
        state = _initial_state("Plan a trip to Tokyo")
        state["errors"] = ["older error"]
        state["delegation"] = {"id": "call_9", "args": {"task": "focus on budget options"}}
        with (
            patch.object(module, "get_llm", lambda **_: None),
            patch.object(module, "run_react_loop", _fake_react("X", ["new error"])),
        ):
            update = await getattr(module, node)(state)
        assert update[result_key] == "X"
        assert update["errors"] == ["new error"]
        assert update["completed_agents"] == [node]
        assert update["llm_calls"] == 1  # delta, not a running total
        (message,) = update["messages"]
        assert (message.tool_call_id, message.name, message.content) == ("call_9", node, "X")

    @pytest.mark.asyncio
    async def test_agent_run_without_a_delegation_adds_no_tool_message(self):
        with (
            patch.object(flight_mod, "get_llm", lambda **_: None),
            patch.object(flight_mod, "run_react_loop", _fake_react("X")),
        ):
            update = await flight_mod.flight_agent(_initial_state("Flights to Tokyo"))
        assert update["messages"] == []
        assert update["flight_results"] == "X"


class TestRouteAfterOrchestrator:
    def test_answer_ends_the_run(self):
        state = _initial_state("q")
        state["messages"].append(AIMessage(content="FINAL PLAN"))
        assert route_after_orchestrator(state) == END

    def test_tool_calls_fan_out_as_sends(self):
        state = _initial_state("q")
        calls = [tool_call("flight_agent", task="find flights"), tool_call("weather_agent", city="Tokyo")]
        state["messages"].append(AIMessage(content="", tool_calls=calls))
        sends = route_after_orchestrator(state)
        assert all(isinstance(send, Send) for send in sends)
        assert [send.node for send in sends] == ["flight_agent", "weather_agent"]
        # Each sub-agent gets the state plus its own delegation
        assert sends[0].arg["delegation"] == {"id": calls[0]["id"], "args": {"task": "find flights"}}
        assert sends[1].arg["delegation"]["args"] == {"city": "Tokyo"}
        assert sends[0].arg["user_query"] == "q"
        assert state["delegation"] == {}  # the shared state is not mutated


class TestSelectDelegations:
    def _names(self, calls, completed=()):
        return [c["name"] for c in _select_delegations(calls, set(completed))]

    def test_unknown_duplicate_and_completed_calls_are_dropped(self):
        calls = [tool_call(n) for n in ("flight_agent", "bogus", "hotel_agent", "flight_agent", "weather_agent")]
        assert self._names(calls, completed=["weather_agent"]) == ["flight_agent", "hotel_agent"]

    def test_itinerary_is_held_back_behind_research(self):
        calls = [tool_call("itinerary_agent"), tool_call("flight_agent")]
        assert self._names(calls) == ["flight_agent"]

    def test_itinerary_alone_is_honoured(self):
        assert self._names([tool_call("itinerary_agent")]) == ["itinerary_agent"]

    def test_ids_and_args_are_kept_or_repaired(self):
        kept, repaired = _select_delegations([
            tool_call("flight_agent", task="t"),
            {"name": "hotel_agent", "args": None, "id": None},
        ], set())
        assert kept["args"] == {"task": "t"} and kept["id"].startswith("call_flight_agent")
        assert repaired["args"] == {} and repaired["id"]

    def test_nothing_requested(self):
        assert _select_delegations(None, set()) == []


class TestRequiredDelegations:
    def _state(self, intent, **extra):
        state = build_initial_state("q")
        state["user_intent"] = intent
        state.update(extra)
        return state

    @pytest.mark.parametrize("intent,expected", [
        ("weather", ["weather_agent"]),
        ("flight", ["flight_agent"]),
        ("full_trip", RESEARCH),
        ("general", []),
    ])
    def test_answering_with_nothing_gathered_gets_the_default_plan(self, intent, expected):
        assert _required_delegations(self._state(intent), decided=True) == expected

    @pytest.mark.parametrize("intent", ["weather", "flights", "flight", "hotel", "general"])
    def test_single_topic_with_data_may_answer(self, intent):
        state = self._state(intent, weather_results="sunny", completed_agents=["weather_agent"])
        assert _required_delegations(state, decided=True) == []

    @pytest.mark.parametrize("intent", ["full_trip", "itinerary"])
    def test_trip_intent_with_research_gets_itinerary(self, intent):
        state = self._state(intent, flight_results="F", completed_agents=["flight_agent"])
        assert _required_delegations(state, decided=True) == ["itinerary_agent"]

    def test_trip_intent_with_itinerary_may_answer(self):
        state = self._state("full_trip", flight_results="F", itinerary="I",
                            completed_agents=["flight_agent", "itinerary_agent"])
        assert _required_delegations(state, decided=True) == []

    def test_undecided_model_gets_the_rest_of_the_default_plan(self):
        state = self._state("full_trip", flight_results="F", completed_agents=["flight_agent"])
        assert _required_delegations(state, decided=False) == ["hotel_agent", "weather_agent"]


class TestParallelDelegation:
    @pytest.mark.asyncio
    async def test_research_agents_run_concurrently(self):
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

        llm = ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"])
        visited, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            llm,
            react={"flight_agent": gated("flight"), "hotel_agent": gated("hotel")},
        )

        assert sorted(started) == ["flight", "hotel"]
        assert state["flight_results"] == "FLIGHT"
        assert state["hotel_results"] == "HOTEL"
        # One fan-in per round: the orchestrator runs before the batch, after it, and after the itinerary.
        assert visited.count("orchestrator") == 3
        assert len(llm.calls) == 3
        assert visited == [
            *INTAKE, "orchestrator",
            *visited[4:7],
            "orchestrator", "itinerary_agent", "orchestrator",
        ]
        assert sorted(visited[4:7]) == RESEARCH

    @pytest.mark.asyncio
    async def test_parallel_writes_are_merged(self):
        _, state = await _run_graph(
            "Plan a 5-day trip to Tokyo",
            ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"]),
        )
        assert sorted(state["completed_agents"]) == sorted(SUBAGENT_NAMES)
        # intent(1) + orchestrator(3) + flight(1) + hotel(1) + itinerary(1)
        assert state["llm_calls"] == 7
        assert state["orchestrator_rounds"] == 2

    @pytest.mark.asyncio
    async def test_completed_agent_is_not_run_again(self):
        llm = ScriptedOrchestratorLLM(["weather_agent"], ["weather_agent"], ["weather_agent"])
        visited, state = await _run_graph("What is the weather in Tokyo?", llm)
        assert visited.count("weather_agent") == 1
        assert visited[-1] == "orchestrator"
        assert state["messages"][-1].content == "FINAL PLAN"

    @pytest.mark.asyncio
    async def test_one_agent_per_turn_still_works(self):
        """An orchestrator that delegates sequentially stays within the round bound."""
        llm = ScriptedOrchestratorLLM(["flight_agent"], ["hotel_agent"], ["weather_agent"])
        visited, state = await _run_graph("Plan a 5-day trip to Tokyo", llm)
        for agent in SUBAGENT_NAMES:
            assert visited.count(agent) == 1
        assert visited[-1] == "orchestrator"
        assert state["messages"][-1].content == "FINAL PLAN"


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
            ScriptedOrchestratorLLM(RESEARCH, ["itinerary_agent"]),
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
