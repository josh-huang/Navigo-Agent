"""Weather agent: fetches current weather + forecast via OpenWeatherMap MCP."""

import asyncio
import logging
from datetime import UTC, date, datetime

from langchain_core.messages import AIMessage

from navigo_agent.graph.slots import TripSlots
from navigo_agent.mcp import (
    extract_destination,
    extract_mcp_text,
    forecast_mcp_search,
    weather_mcp_search,
)
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

WEATHER_TIMEOUT_SECONDS = 25
FORECAST_DAYS = 5  # OpenWeather's free forecast horizon


def forecast_coverage_note(start_date: str | None, today: date | None = None) -> str:
    """Say plainly when the forecast does not reach the travel dates.

    Live weather is only meaningful for trips starting within the forecast
    window. For a trip months away, presenting today's weather as "the
    weather for your trip" would mislead both the user and the itinerary
    agent, so the note tells downstream prompts how to treat the data.
    """
    if not start_date:
        return ""
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return ""
    today = today or datetime.now(UTC).date()
    days_ahead = (start - today).days
    if days_ahead <= FORECAST_DAYS:
        return ""
    return (
        f"\n\nNote: the trip starts on {start.isoformat()}, {days_ahead} days from now — beyond this "
        f"{FORECAST_DAYS}-day forecast. The data above shows current conditions only and is NOT a "
        "forecast for the travel dates. Describe the typical seasonal climate for that time of "
        "year instead, and say that an accurate forecast is not available yet."
    )


async def weather_agent(state: TravelState) -> dict:
    """Fetch current weather + 5-day forecast for the destination city.

    Destination resolution:
      1. Use the extracted trip slots (destination_city, then destination)
      2. Fall back to LLM-based extract_destination() which parses the user query

    Runs in parallel with the flight/hotel agents, so it returns deltas only
    and always reports itself completed — on success *and* failure.
    """
    slots = TripSlots(**(state.get("trip") or {}))
    city = slots.weather_city or state.get("extracted_destination")
    llm_calls = 0

    try:
        if not city:
            # Sync LLM call — keep it off the event loop so parallel agents aren't blocked.
            city = await asyncio.to_thread(extract_destination, state["user_query"])
            llm_calls = 1
        logger.info("Weather agent: fetching weather for '%s'", city)
        weather_data, forecast_data = await asyncio.wait_for(
            asyncio.gather(weather_mcp_search(city), forecast_mcp_search(city)),
            timeout=WEATHER_TIMEOUT_SECONDS,
        )
        # Unwrap MCP content-block format → plain text
        current_text = extract_mcp_text(weather_data)
        forecast_text = extract_mcp_text(forecast_data)
        weather_text = f"""Current Weather:
{current_text}

Forecast:
{forecast_text}{forecast_coverage_note(slots.start_date)}"""
    except Exception as e:  # noqa: BLE001
        if isinstance(e, TimeoutError):
            e = TimeoutError(f"no response within {WEATHER_TIMEOUT_SECONDS}s")
        logger.error("Weather agent failed: %s", e)
        weather_text = f"Weather information unavailable: {e!s}"
        # Error path: still mark completed and save the extracted destination for reuse
        return {
            "weather_results": weather_text,
            "messages": [AIMessage(content="Weather check encountered an error.")],
            "llm_calls": llm_calls,
            "errors": [f"weather_agent: {e!s}"],
            "completed_agents": ["weather_agent"],
        }

    return {
        "weather_results": weather_text,
        "messages": [AIMessage(content="Weather information fetched.")],
        "llm_calls": llm_calls,
        "completed_agents": ["weather_agent"],
    }
