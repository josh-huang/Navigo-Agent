"""Itinerary agent: synthesizes collected data into a day-by-day plan."""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

from navigo_agent.config import get_llm
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)


async def itinerary_agent(state: TravelState) -> dict:
    """Generate day-by-day itinerary from flight, hotel, and weather data."""
    logger.info("Itinerary agent: generating plan for '%s'", state["user_query"][:80])

    prompt = f"""
Create a complete travel itinerary based on the following information:
User Query: {state['user_query']}
Flight Results: {state.get('flight_results', 'No flight data available.')}
Hotel Results: {state.get('hotel_results', 'No hotel data available.')}
Weather Results: {state.get('weather_results', 'No weather data available.')}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    try:
        llm = get_llm(temperature=0.5)  # slightly higher temp for creative planning
        response = await llm.ainvoke([
            SystemMessage(content="You are a travel assistant that creates detailed itineraries based on user queries and search results."),
            HumanMessage(content=prompt),
        ])
        itinerary_text = response.content
    except Exception as e:
        logger.error("Itinerary agent failed: %s", e)
        return {
            "itinerary": f"Itinerary generation failed: {str(e)}",
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"itinerary_agent: {str(e)}"],
            "completed_agents": list(set(state.get("completed_agents", []) + ["itinerary_agent"])),
        }

    completed = list(set(state.get("completed_agents", []) + ["itinerary_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "itinerary_agent"]

    return {
        "itinerary": itinerary_text,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "completed_agents": completed,
        "pending_agents": pending,
    }
