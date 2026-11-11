"""Navigo-Agent: AI-powered multi-agent travel planner.

Built on LangGraph with supervisor routing, MCP tool integration,
and production-grade guardrails.

Usage:
    from navigo_agent import run_travel_agent

    result = await run_travel_agent("Plan a 7-day Japan trip")
    print(result["answer"])
"""

import logging
import uuid

from navigo_agent.middleware import setup_middleware
from navigo_agent.state import TravelState, build_initial_state

logger = logging.getLogger(__name__)

__all__ = [
    "TravelState",
    "run_travel_agent",
    "setup_middleware",
]


async def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict:
    """Run the travel planning graph with the given user input.

    Args:
        user_input: Natural language travel request.
        thread_id: Optional thread ID for session continuity (checkpointing).
                   If not provided, a new thread is created.

    Returns:
        dict with keys: thread_id, answer, flight_results, hotel_results,
                        weather_results, itinerary, llm_calls
    """
    # Always create a fresh thread for each invocation.
    # Reusing a completed thread makes every reducer-backed field (messages,
    # errors, llm_calls, completed_agents) carry over from the previous run —
    # e.g. agents would look "already completed" and never be dispatched.
    # Multi-turn support can be added later with proper state trimming.
    thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    from navigo_agent.graph.builder import get_compiled_graph

    graph = get_compiled_graph()

    try:
        result = await graph.ainvoke(
            build_initial_state(user_input),
            config=config,
        )
    except Exception:
        logger.exception("Graph invocation failed for thread %s", thread_id)
        raise

    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else "No response generated."

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
