"""Weather agent: fetches current weather + forecast via OpenWeatherMap MCP."""

import asyncio
import logging
from datetime import UTC, date, datetime

from navigo_agent.graph.agents.delegation import delegation_args, report
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
      1. The extracted trip slots (destination_city, then destination) — the
         slot extractor resolved them specifically for a weather lookup
      2. The city the orchestrator named when it delegated
      3. LLM-based extract_destination() over the user query

    Runs in parallel with the flight/hotel sub-agents, so it returns deltas
    only and always reports back — on success *and* failure.
    """
    slots = TripSlots(**(state.get("trip") or {}))
    delegated_city = str(delegation_args(state).get("city") or "").strip()
    city = slots.weather_city or delegated_city or state.get("extracted_destination")
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
        # Error path: still report back, so the orchestrator carries on without the data.
        return report(
            state, "weather_agent", f"Weather information unavailable: {e!s}",
            llm_calls=llm_calls, errors=[f"weather_agent: {e!s}"],
        )

    return report(state, "weather_agent", weather_text, llm_calls=llm_calls)
