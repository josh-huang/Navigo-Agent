"""The orchestrator's request brief (answer shaping), daily forecast summary, CORS."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from navigo_agent.graph.orchestrator import (
    build_orchestrator_messages,
    build_request_brief,
)
from navigo_agent.mcp import weather_server
from navigo_agent.middleware import configure_cors
from navigo_agent.state import build_initial_state


def _state(intent, **results):
    state = build_initial_state("some question")
    state["user_intent"] = intent
    state.update(results)
    return state


class TestRequestBrief:
    @pytest.mark.parametrize("intent", ["weather", "flights", "flight", "hotel", "general"])
    def test_single_topic_gets_focused_answer(self, intent):
        brief = build_request_brief(_state(intent, weather_results="Sunny, 22C", completed_agents=["weather_agent"]))
        assert f"- Intent: {intent}" in brief
        assert "Do NOT add sections" in brief
        # No plan template for a single-topic question
        assert "Day-by-Day Itinerary" not in brief
        assert "Trip Summary" not in brief
        # Reports travel as ToolMessages — the brief only tracks who has reported
        assert "Sunny, 22C" not in brief
        assert "Reports received from: weather_agent" in brief
        assert "Sub-agents not used yet: flight_agent, hotel_agent, itinerary_agent" in brief

    @pytest.mark.parametrize("intent", ["full_trip", "itinerary"])
    def test_trip_gets_full_plan_and_flags_missing_data(self, intent):
        brief = build_request_brief(_state(intent, flight_results="ANA", itinerary="Day 1"))
        assert "1. Trip Summary" in brief
        assert "5. Day-by-Day Itinerary" in brief
        assert "say the information was unavailable rather than fabricating it" in brief
        assert "Do NOT add sections" not in brief

    def test_progress_and_problems_are_reported(self):
        state = _state("full_trip", errors=["hotel_agent(search): timed out after 30s"], orchestrator_rounds=1)
        brief = build_request_brief(state)
        assert "Reports received from: none yet" in brief
        assert "Delegation rounds used: 1 of 3" in brief
        assert "Problems so far: hotel_agent(search): timed out after 30s" in brief
        assert "Problems so far: none" in build_request_brief(_state("general"))

    def test_conversation_is_system_brief_then_transcript(self):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        state = _state("weather")
        call = {"name": "weather_agent", "args": {"city": "Tokyo"}, "id": "c1", "type": "tool_call"}
        state["messages"] += [
            HumanMessage(content="a clarification reply"),      # user turns are in the brief, not repeated
            AIMessage(content="", tool_calls=[call]),
            ToolMessage(content="Sunny, 22C", tool_call_id="c1", name="weather_agent"),
        ]
        system, brief, delegation, report = build_orchestrator_messages(state)
        assert isinstance(system, SystemMessage)
        for agent in ("flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"):
            assert f"- {agent}: " in system.content
        assert isinstance(brief, HumanMessage) and "## User Request" in brief.content
        assert delegation.tool_calls[0]["name"] == "weather_agent"
        assert report.content == "Sunny, 22C"


def _entry(dt, temp, lo, hi, desc):
    return {"dt_txt": dt, "main": {"temp": temp, "temp_min": lo, "temp_max": hi},
            "weather": [{"description": desc}]}


class TestForecastSummary:
    def test_groups_by_day(self):
        entries = [
            _entry("2026-12-01 09:00:00", 10, 9, 11, "light rain"),
            _entry("2026-12-01 12:00:00", 14, 13, 15.44, "light rain"),
            _entry("2026-12-01 15:00:00", 13, 12, 14, "clear sky"),
            _entry("2026-12-02 09:00:00", 5, 4.26, 6, "snow"),
        ]
        assert weather_server.summarise_forecast(entries) == [
            {"date": "2026-12-01", "temp_min_c": 9, "temp_max_c": 15.4, "condition": "light rain"},
            {"date": "2026-12-02", "temp_min_c": 4.3, "temp_max_c": 6, "condition": "snow"},
        ]

    def test_skips_malformed_entries(self):
        entries = [{"dt_txt": "2026-12-01 09:00:00"}, {}, _entry("2026-12-03 09:00:00", 1, 0, 2, "fog")]
        assert [d["date"] for d in weather_server.summarise_forecast(entries)] == ["2026-12-03"]

    def test_get_forecast_covers_all_days_and_uses_timeout(self):
        payload = {"list": [_entry(f"2026-12-0{d} 12:00:00", 10, 8, 12, "clouds") for d in range(1, 6)]}
        response = MagicMock(status_code=200, json=lambda: payload)
        with patch.object(weather_server.requests, "get", return_value=response) as get:
            result = weather_server.get_forecast("Tokyo")
        assert result["city"] == "Tokyo"
        assert len(result["daily"]) == 5
        assert get.call_args.kwargs["timeout"] == weather_server.REQUEST_TIMEOUT_SECONDS

    def test_get_forecast_passes_api_error_through(self):
        response = MagicMock(status_code=404, json=lambda: {"cod": "404", "message": "city not found"})
        with patch.object(weather_server.requests, "get", return_value=response):
            assert weather_server.get_forecast("Atlantis") == {"cod": "404", "message": "city not found"}

    def test_current_weather_uses_timeout(self):
        response = MagicMock(status_code=401, json=lambda: {"cod": 401})
        with patch.object(weather_server.requests, "get", return_value=response) as get:
            weather_server.get_current_weather("Tokyo")
        assert get.call_args.kwargs["timeout"] == weather_server.REQUEST_TIMEOUT_SECONDS


class TestCors:
    def _preflight(self, monkeypatch, allowed):
        if allowed is None:
            monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("ALLOWED_ORIGINS", allowed)
        app = FastAPI()
        configure_cors(app)
        return TestClient(app).options("/x", headers={
            "Origin": "https://app.example.com", "Access-Control-Request-Method": "POST",
        })

    def test_wildcard_origin_never_allows_credentials(self, monkeypatch):
        response = self._preflight(monkeypatch, None)
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in response.headers

    def test_explicit_origins_allow_credentials(self, monkeypatch):
        response = self._preflight(monkeypatch, "https://app.example.com")
        assert response.headers["access-control-allow-origin"] == "https://app.example.com"
        assert response.headers["access-control-allow-credentials"] == "true"
