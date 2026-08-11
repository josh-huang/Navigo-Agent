"""Weather agent: fetches current weather + forecast via OpenWeatherMap MCP."""

import logging
from langchain_core.messages import AIMessage

from navigo_agent.state import TravelState
from mcp_client import weather_mcp_search, forecast_mcp_search, extract_destination

logger = logging.getLogger(__name__)


async def weather_agent(state: TravelState) -> dict:
    """Fetch current weather + 5-day forecast for the destination city."""
    city = state.get("extracted_destination") or extract_destination(state["user_query"])
    logger.info("Weather agent: fetching weather for '%s'", city)

    try:
        weather_data = await weather_mcp_search(city)
        forecast_data = await forecast_mcp_search(city)
        weather_text = f"""Current Weather:
{weather_data}

Forecast:
{forecast_data}"""
    except Exception as e:
        logger.error("Weather agent failed: %s", e)
        weather_text = f"Weather information unavailable: {str(e)}"
        return {
            "weather_results": weather_text,
            "extracted_destination": city,
            "messages": [AIMessage(content="Weather check encountered an error.")],
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"weather_agent: {str(e)}"],
            "completed_agents": list(set(state.get("completed_agents", []) + ["weather_agent"])),
        }

    completed = list(set(state.get("completed_agents", []) + ["weather_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "weather_agent"]

    return {
        "weather_results": weather_text,
        "extracted_destination": city,
        "messages": [AIMessage(content="Weather information fetched.")],
        "completed_agents": completed,
        "pending_agents": pending,
    }
