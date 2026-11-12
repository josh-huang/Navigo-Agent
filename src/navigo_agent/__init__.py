"""Navigo-Agent: AI-powered multi-agent travel planner.

Built on LangGraph with supervisor routing, MCP tool integration,
and production-grade guardrails.

Usage:
    from navigo_agent import run_travel_agent

    result = await run_travel_agent("Plan a 7-day Japan trip")
    print(result["answer"])
"""

import logging
import time

from navigo_agent.middleware import setup_middleware
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

__all__ = [
    "TravelState",
    "run_travel_agent",
    "setup_middleware",
]


async def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    *,
    allow_clarification: bool = True,
) -> dict:
    """Run the travel planning graph with the given user input.

    Args:
        user_input: Natural language travel request — or, when `thread_id`
                    refers to a run paused on a clarification question, the
                    user's reply to that question.
        thread_id: Thread returned by a previous call. A paused thread is
                   resumed; a finished one makes this message a follow-up
                   of that conversation (run on a new thread).
        allow_clarification: False → never pause to ask the user (batch/eval).

    Returns:
        dict with keys: thread_id, answer, flight_results, hotel_results,
                        weather_results, itinerary, llm_calls,
                        needs_clarification, question, missing.
        When needs_clarification is True, `answer` is the question and the
        caller should send the user's reply with the same thread_id.
    """
    from navigo_agent.graph.builder import get_compiled_graph
    from navigo_agent.session import (
        clarification_from_result,
        log_run_summary,
        plan_run,
    )

    graph = get_compiled_graph()
    plan = await plan_run(graph, user_input, thread_id, allow_clarification=allow_clarification)

    started = time.perf_counter()
    try:
        result = await graph.ainvoke(plan.graph_input, config=plan.config)
    except Exception:
        logger.exception("Graph invocation failed for thread %s", plan.thread_id)
        log_run_summary(plan.thread_id, None, started, outcome="failed")
        raise

    clarification = clarification_from_result(result)
    log_run_summary(
        plan.thread_id, result, started,
        outcome="awaiting_clarification" if clarification else "completed",
    )
    if clarification:
        return {
            "thread_id": plan.thread_id,
            "answer": clarification["question"],
            "needs_clarification": True,
            "question": clarification["question"],
            "missing": clarification["missing"],
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": result.get("llm_calls", 0),
        }

    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else "No response generated."

    return {
        "thread_id": plan.thread_id,
        "answer": final_answer,
        "needs_clarification": False,
        "follow_up": bool(result.get("history")),
        "question": None,
        "missing": [],
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
