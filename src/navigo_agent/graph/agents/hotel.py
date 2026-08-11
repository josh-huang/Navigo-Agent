"""Hotel agent: searches Tavily for hotel recommendations."""

import logging
from langchain_core.messages import AIMessage

from navigo_agent.state import TravelState
from mcp_client import tavily_mcp_search

logger = logging.getLogger(__name__)


async def hotel_agent(state: TravelState) -> dict:
    """Search hotels via Tavily MCP."""
    query = f"Best hotels for {state['user_query']}"
    logger.info("Hotel agent: searching '%s'", query[:80])

    try:
        hotel_data = await tavily_mcp_search(query)
    except Exception as e:
        logger.error("Hotel agent failed: %s", e)
        hotel_data = f"Hotel information unavailable: {str(e)}"
        return {
            "hotel_results": hotel_data,
            "messages": [AIMessage(content="Hotel search encountered an error.")],
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"hotel_agent: {str(e)}"],
            "completed_agents": list(set(state.get("completed_agents", []) + ["hotel_agent"])),
        }

    completed = list(set(state.get("completed_agents", []) + ["hotel_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "hotel_agent"]

    return {
        "hotel_results": hotel_data,
        "messages": [AIMessage(content="Hotel results fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "completed_agents": completed,
        "pending_agents": pending,
    }
