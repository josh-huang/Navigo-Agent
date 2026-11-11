"""Shared fixtures for Navigo-Agent tests."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def mock_llm_response():
    """Return a mock LLM response object with .content attribute."""

    def _make(content: str = "Mock LLM response"):
        mock = MagicMock()
        mock.content = content
        return mock

    return _make


@pytest.fixture
def mock_llm():
    """Return an AsyncMock for the LLM's ainvoke method."""

    async def _mock_ainvoke(messages):
        mock = MagicMock()
        mock.content = "Mock LLM response to: " + str(messages[-1].content)[:50]
        return mock

    return AsyncMock(side_effect=_mock_ainvoke)


@pytest.fixture
def sample_state():
    """Return a minimal TravelState dict for testing."""
    return {
        "messages": [],
        "user_query": "Plan a trip to Tokyo",
        "user_intent": "full_trip",
        "trip": {"destination": "Tokyo"},
        "extracted_destination": "Tokyo",
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


# ── Real graph with mocked LLMs/tools ─────────────────────────────────

class _SupervisorLLM:
    """Dispatches the three data agents on the first call of each run, then finalizes."""

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    async def ainvoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        agents = (
            ["flight_agent", "hotel_agent", "weather_agent"]
            if "Completed agents: none" in prompt else []
        )
        return AIMessage(content=json.dumps({
            "reasoning": "scripted", "next_action": "x",
            "agents_to_dispatch": agents, "is_complete": not agents,
        }))


class _FinalLLM:
    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=f"FINAL PLAN #{len(self.prompts)}" if len(self.prompts) > 1 else "FINAL PLAN")


@pytest.fixture
def graph_env():
    """Real graph + in-memory checkpointer; LLMs/tools mocked.

    Keys:
      extract — mock of the first-turn trip extractor: returns (intent, slots)
      follow  — mock of the follow-up extractor: returns (intent, slots, is_follow_up)
      weather — mock of the weather tool
      supervisor_prompts / final_prompts — prompts those LLMs received
      agent_messages — {agent: [first human message of each run]}
    """
    import navigo_agent.graph.agents.final as final_mod
    import navigo_agent.graph.agents.flight as flight_mod
    import navigo_agent.graph.agents.hotel as hotel_mod
    import navigo_agent.graph.agents.itinerary as itinerary_mod
    import navigo_agent.graph.agents.weather as weather_mod
    import navigo_agent.graph.builder as builder_mod
    import navigo_agent.graph.routing as routing_mod
    import navigo_agent.graph.supervisor as supervisor_mod
    from navigo_agent import streaming

    extract, follow = AsyncMock(), AsyncMock()
    weather = AsyncMock(return_value="sunny")
    supervisor_prompts: list[str] = []
    final_prompts: list[str] = []
    agent_messages: dict[str, list[str]] = {}

    def fake_react(answer):
        async def _run(**kwargs):
            agent_messages.setdefault(kwargs["agent_name"], []).append(kwargs["conversation"][1].content)
            return answer, 1, []

        return _run

    saved = builder_mod._compiled_graph
    with (
        patch.object(routing_mod, "extract_trip_request", extract),
        patch.object(routing_mod, "extract_follow_up", follow),
        patch.object(supervisor_mod, "get_llm", lambda **_: _SupervisorLLM(supervisor_prompts)),
        patch.object(final_mod, "get_llm", lambda **_: _FinalLLM(final_prompts)),
        patch.object(flight_mod, "get_llm", lambda **_: None),
        patch.object(hotel_mod, "get_llm", lambda **_: None),
        patch.object(itinerary_mod, "get_llm", lambda **_: None),
        patch.object(flight_mod, "run_react_loop", fake_react("FLIGHTS")),
        patch.object(hotel_mod, "run_react_loop", fake_react("HOTELS")),
        patch.object(itinerary_mod, "run_react_loop", fake_react("ITINERARY")),
        patch.object(weather_mod, "weather_mcp_search", weather),
        patch.object(weather_mod, "forecast_mcp_search", AsyncMock(return_value="mild")),
    ):
        graph = build_graph_with_memory()
        builder_mod._compiled_graph = graph
        with patch.object(streaming, "get_compiled_graph", lambda: graph):
            try:
                yield {
                    "graph": graph, "extract": extract, "follow": follow, "weather": weather,
                    "supervisor_prompts": supervisor_prompts, "final_prompts": final_prompts,
                    "agent_messages": agent_messages,
                }
            finally:
                builder_mod._compiled_graph = saved


def build_graph_with_memory():
    from navigo_agent.graph.builder import build_travel_graph

    return build_travel_graph().compile(checkpointer=MemorySaver())
