"""Shared fixtures for Navigo-Agent tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from tests.fakes import SensibleOrchestratorLLM


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Each test starts with an empty rate-limit bucket."""
    from navigo_agent.middleware import limiter

    limiter.reset()
    yield
    limiter.reset()


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
        "orchestrator_rounds": 0,
        "delegation": {},
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "completed_agents": [],
        "llm_calls": 0,
        "errors": [],
    }


# ── Real graph with mocked LLMs/tools ─────────────────────────────────

@pytest.fixture
def graph_env():
    """Real graph + in-memory checkpointer; LLMs/tools mocked.

    Keys:
      extract — mock of the first-turn trip extractor: returns (intent, slots)
      follow  — mock of the follow-up extractor: returns (intent, slots, is_follow_up)
      weather — mock of the weather tool
      orchestrator_prompts — the request brief of every orchestrator LLM call
      answer_prompts — the briefs of the calls that produced a final answer
      agent_messages — {agent: [first human message of each run]}
    """
    import navigo_agent.graph.agents.flight as flight_mod
    import navigo_agent.graph.agents.hotel as hotel_mod
    import navigo_agent.graph.agents.itinerary as itinerary_mod
    import navigo_agent.graph.agents.weather as weather_mod
    import navigo_agent.graph.builder as builder_mod
    import navigo_agent.graph.intake as intake_mod
    import navigo_agent.graph.orchestrator as orchestrator_mod
    from navigo_agent import streaming

    extract, follow = AsyncMock(), AsyncMock()
    weather = AsyncMock(return_value="sunny")
    orchestrator_prompts: list[str] = []
    answer_prompts: list[str] = []
    agent_messages: dict[str, list[str]] = {}

    def fake_react(answer):
        async def _run(**kwargs):
            agent_messages.setdefault(kwargs["agent_name"], []).append(kwargs["conversation"][1].content)
            return answer, 1, []

        return _run

    saved = builder_mod._compiled_graph
    with (
        patch.object(intake_mod, "extract_trip_request", extract),
        patch.object(intake_mod, "extract_follow_up", follow),
        patch.object(
            orchestrator_mod, "get_llm",
            lambda **_: SensibleOrchestratorLLM(orchestrator_prompts, answer_prompts),
        ),
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
                    "orchestrator_prompts": orchestrator_prompts, "answer_prompts": answer_prompts,
                    "agent_messages": agent_messages,
                }
            finally:
                builder_mod._compiled_graph = saved


def build_graph_with_memory():
    from navigo_agent.graph.builder import build_travel_graph

    return build_travel_graph().compile(checkpointer=MemorySaver())
