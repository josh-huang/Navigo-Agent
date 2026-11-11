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
from collections.abc import AsyncGenerator

from navigo_agent.graph.builder import get_compiled_graph
from navigo_agent.guardrails import guard_output
from navigo_agent.session import pending_clarification, plan_run

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
      - `progress`: an agent node has started or completed (`status`: started | completed)
      - `agent_result`: an agent has finished (includes partial results)
      - `complete`: the entire graph run completed
      - `clarification`: the run paused to ask the user a question;
        send the reply with the returned thread_id to continue
      - `error`: a fatal error occurred

    Args:
        user_input: The user's travel request.
        thread_id: A paused thread is resumed; a finished one makes this message
                   a follow-up of that conversation (run on a new thread).

    Yields:
        SSE-formatted strings.
    """
    graph = get_compiled_graph()
    # Resume `thread_id` if it is paused on a clarification question;
    # otherwise a new run (a follow-up when `thread_id` is a finished turn).
    plan = await plan_run(graph, user_input, thread_id)
    thread_id = plan.thread_id
    config = plan.config

    running: set[str] = set()  # nodes currently executing (several at once when parallel)
    complete_payload: dict | None = None

    try:
        async for event in graph.astream_events(plan.graph_input, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Node start — lets the UI show which agents are working right now.
            # (Inner runnables reuse the node name; `running` de-duplicates them.)
            if kind == "on_chain_start" and name in AGENT_DISPLAY_NAMES and name not in running:
                running.add(name)
                yield _sse_event("progress", {
                    "agent": name,
                    "display": AGENT_DISPLAY_NAMES[name],
                    "icon": AGENT_ICONS.get(name, "⏳"),
                    "status": "started",
                })

            # Node completion events
            if kind == "on_chain_end" and name in AGENT_DISPLAY_NAMES and name in running:
                running.discard(name)
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
                        result_data["flight_results"] = guard_output(output["flight_results"][:500])
                    elif name == "hotel_agent" and output.get("hotel_results"):
                        result_data["hotel_results"] = guard_output(output["hotel_results"][:500])
                    elif name == "weather_agent" and output.get("weather_results"):
                        result_data["weather_results"] = guard_output(output["weather_results"][:500])
                    elif name == "itinerary_agent" and output.get("itinerary"):
                        result_data["itinerary"] = guard_output(output["itinerary"][:500])

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

                    # Held back until the stream ends: a run that pauses on a
                    # clarification question also emits this event.
                    # Same output guard as the non-streaming endpoint.
                    complete_payload = {
                        "thread_id": thread_id,
                        "answer": guard_output(final_answer),
                        "flight_results": guard_output(output.get("flight_results", "")),
                        "hotel_results": guard_output(output.get("hotel_results", "")),
                        "weather_results": guard_output(output.get("weather_results", "")),
                        "itinerary": guard_output(output.get("itinerary", "")),
                        "llm_calls": output.get("llm_calls", 0),
                        "follow_up": bool(output.get("history")),
                    }

        clarification = await pending_clarification(graph, config)
        if clarification:
            yield _sse_event("clarification", {"thread_id": thread_id, **clarification})
        elif complete_payload is not None:
            yield _sse_event("complete", complete_payload)

    except Exception:
        logger.exception("Streaming graph execution failed for thread %s", thread_id)
        yield _sse_event("error", {
            "message": "An error occurred while generating your travel plan. Please try again.",
            "thread_id": thread_id,
        })
