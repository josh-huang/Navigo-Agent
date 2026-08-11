"""Shared fixtures for Navigo-Agent tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
