"""SSE streaming for the Navigo-Agent supervisor graph.

Provides a streaming wrapper that yields Server-Sent Events as each agent
node completes. Frontend renders these as real-time progress indicators.

The streaming endpoint uses LangGraph's `astream_events()` to pick up
node completions and the final answer.

Usage (in FastAPI route):
    from navigo_agent.streaming import stream_travel_plan
    return StreamingResponse(
        stream_travel_plan(user_input, thread_id),
        media_type="text/event-stream",
    )
"""

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage

from navigo_agent.graph.builder import get_compiled_graph

logger = logging.getLogger(__name__)

# Agent display names for frontend progress indicators
AGENT_DISPLAY_NAMES = {
    "intent_classifier": "Analysing your request",
    "supervisor": "Planning next steps",
    "flight_agent": "Searching flights",
    "hotel_agent": "Finding hotels",
    "weather_agent": "Checking weather",
    "itinerary_agent": "Building itinerary",
    "final_synthesizer": "Finalising your plan",
}

AGENT_ICONS = {
    "intent_classifier": "🤔",
    "supervisor": "🧠",
    "flight_agent": "✈️",
    "hotel_agent": "🏨",
    "weather_agent": "🌤️",
    "itinerary_agent": "🗺️",
    "final_synthesizer": "✨",
}


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_travel_plan(
    user_input: str,
    thread_id: str | None = None,
) -> AsyncGenerator[str]:
    """Stream the travel planning graph execution as SSE events.

    Events emitted:
      - `progress`: an agent node has started or completed
      - `agent_result`: an agent has finished (includes partial results)
      - `complete`: the entire graph run completed
      - `error`: a fatal error occurred

    Args:
        user_input: The user's travel request.
        thread_id: Optional thread ID for session continuity.

    Yields:
        SSE-formatted strings.
    """
    thread_id = thread_id or f"user_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    graph = get_compiled_graph()

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "user_intent": "",
        "extracted_destination": None,
        "supervisor_reasoning": "",
        "supervisor_loop_count": 0,
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "pending_agents": [],
        "completed_agents": [],
        "llm_calls": 0,
        "errors": [],
    }

    completed_nodes: set[str] = set()

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Node completion events
            if kind == "on_chain_end" and name in AGENT_DISPLAY_NAMES and name not in completed_nodes:
                completed_nodes.add(name)
                display_name = AGENT_DISPLAY_NAMES[name]
                icon = AGENT_ICONS.get(name, "⏳")

                yield _sse_event("progress", {
                    "agent": name,
                    "display": display_name,
                    "icon": icon,
                    "status": "completed",
                })

                # Extract agent-specific results from the output
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    result_data = {}
                    if name == "flight_agent" and output.get("flight_results"):
                        result_data["flight_results"] = output["flight_results"][:500]
                    elif name == "hotel_agent" and output.get("hotel_results"):
                        result_data["hotel_results"] = output["hotel_results"][:500]
                    elif name == "weather_agent" and output.get("weather_results"):
                        result_data["weather_results"] = output["weather_results"][:500]
                    elif name == "itinerary_agent" and output.get("itinerary"):
                        result_data["itinerary"] = output["itinerary"][:500]

                    if result_data:
                        yield _sse_event("agent_result", result_data)

            # Final graph completion
            if kind == "on_chain_end" and name == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    messages = output.get("messages", [])
                    final_answer = ""
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "content"):
                            final_answer = last_msg.content
                        elif isinstance(last_msg, dict):
                            final_answer = last_msg.get("content", "")

                    yield _sse_event("complete", {
                        "thread_id": thread_id,
                        "answer": final_answer,
                        "flight_results": output.get("flight_results", ""),
                        "hotel_results": output.get("hotel_results", ""),
                        "weather_results": output.get("weather_results", ""),
                        "itinerary": output.get("itinerary", ""),
                        "llm_calls": output.get("llm_calls", 0),
                    })

    except Exception:
        logger.exception("Streaming graph execution failed for thread %s", thread_id)
        yield _sse_event("error", {
            "message": "An error occurred while generating your travel plan. Please try again.",
            "thread_id": thread_id,
        })
