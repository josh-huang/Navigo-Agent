"""Flight agent: searches AviationStack for airport/airline data, formats with LLM."""

import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from navigo_agent.config import get_llm
from navigo_agent.state import TravelState
from mcp_client import aviation_mcp_call

logger = logging.getLogger(__name__)

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


async def flight_agent(state: TravelState) -> dict:
    """Search flights via AviationStack MCP, format with LLM."""
    logger.info("Flight agent: searching flights for '%s'", state["user_query"][:80])
    query = state["user_query"]

    try:
        airports = await aviation_mcp_call("list_airports")
        airlines = await aviation_mcp_call("list_airlines")

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt),
        ])
        flight_data = response.content

    except Exception as e:
        logger.error("Flight agent failed: %s", e)
        flight_data = f"Flight information unavailable: {str(e)}"
        return {
            "flight_results": flight_data,
            "messages": [AIMessage(content="Flight search encountered an error.")],
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"flight_agent: {str(e)}"],
            "completed_agents": list(set(state.get("completed_agents", []) + ["flight_agent"])),
        }

    # Update tracking
    completed = list(set(state.get("completed_agents", []) + ["flight_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "flight_agent"]

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
        "completed_agents": completed,
        "pending_agents": pending,
    }
