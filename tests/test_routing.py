"""Tests for intent classification and routing logic."""

import pytest
from navigo_agent.graph.routing import _fast_path_classify


class TestIntentClassifier:
    """Fast-path intent classification tests."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is the weather in Tokyo right now?", "weather"),
            ("Tell me the forecast for Paris", "weather"),
            ("Is it going to rain in Bangkok tomorrow?", "weather"),
            ("How hot is Dubai in August?", "weather"),
            ("Find me flights from Dhaka to Dubai", "flights"),
            ("What airlines fly from JFK to London?", "flights"),
            ("I need a one-way flight to Singapore", "flights"),
            ("Find budget hotels in Bangkok", "hotel"),
            ("Best resorts in Maldives for honeymoon", "hotel"),
            ("Airbnb vs hotel in Tokyo which is better?", "hotel"),
        ],
    )
    def test_specific_intent_queries(self, query, expected):
        result = _fast_path_classify(query)
        assert result == expected, f"Expected '{expected}' but got '{result}' for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "Plan a complete 7 days Japan trip from Bangladesh including flights, hotels",
            "I want to visit Thailand for 7 days, help me plan everything",
            "Plan a 5-day trip to Singapore with family, mid-range budget",
            "Travel to Europe for 2 weeks on a budget",
        ],
    )
    def test_full_trip_queries(self, query):
        result = _fast_path_classify(query)
        assert result == "full_trip", f"Expected 'full_trip' but got '{result}' for: {query}"

    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about Tokyo",
            "What should I know before visiting Japan?",
            "Hello",
        ],
    )
    def test_ambiguous_returns_none(self, query):
        result = _fast_path_classify(query)
        assert result is None, f"Expected None but got '{result}' for: {query}"
