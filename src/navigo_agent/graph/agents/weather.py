"""Weather agent: fetches current weather + forecast via OpenWeatherMap MCP."""

import logging

from langchain_core.messages import AIMessage

from navigo_agent.mcp import (
    extract_destination,
    extract_mcp_text,
    forecast_mcp_search,
    weather_mcp_search,
)
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)


async def weather_agent(state: TravelState) -> dict:
    """Fetch current weather + 5-day forecast for the destination city.

    Destination resolution:
      1. Use extracted_destination if already set (by a prior agent or intent classifier)
      2. Fall back to LLM-based extract_destination() which parses the user query

    Returns a state update with weather_results, extracted_destination for reuse,
    and completed/pending agent tracking.
    """
    city = state.get("extracted_destination") or extract_destination(state["user_query"])
    logger.info("Weather agent: fetching weather for '%s'", city)

    try:
        weather_data = await weather_mcp_search(city)
        forecast_data = await forecast_mcp_search(city)
        # Unwrap MCP content-block format → plain text
        current_text = extract_mcp_text(weather_data)
        forecast_text = extract_mcp_text(forecast_data)
        weather_text = f"""Current Weather:
{current_text}

Forecast:
{forecast_text}"""
    except Exception as e:  # noqa: BLE001
        logger.error("Weather agent failed: %s", e)
        weather_text = f"Weather information unavailable: {e!s}"
        # Error path: still mark completed and save the extracted destination for reuse
        return {
            "weather_results": weather_text,
            "extracted_destination": city,
            "messages": [AIMessage(content="Weather check encountered an error.")],
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"weather_agent: {e!s}"],
            "completed_agents": list(set(state.get("completed_agents", []) + ["weather_agent"])),
        }

    # Success path: remove self from pending, add to completed
    completed = list(set(state.get("completed_agents", []) + ["weather_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "weather_agent"]

    return {
        "weather_results": weather_text,
        "extracted_destination": city,  # cache for other agents to reuse
        "messages": [AIMessage(content="Weather information fetched.")],
        "completed_agents": completed,
        "pending_agents": pending,
    }
