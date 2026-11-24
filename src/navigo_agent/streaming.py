"""SSE streaming for the Navigo-Agent orchestrator graph.

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
import time
from collections.abc import AsyncGenerator

from navigo_agent.graph.agents.delegation import SUBAGENTS
from navigo_agent.graph.builder import get_compiled_graph
from navigo_agent.guardrails import guard_output
from navigo_agent.session import log_run_summary, pending_clarification, plan_run

logger = logging.getLogger(__name__)

# Agent display names for frontend progress indicators
AGENT_DISPLAY_NAMES = {
    "intent_classifier": "Analysing your request",
    "orchestrator": "Orchestrating your trip",
    "flight_agent": "Searching flights",
    "hotel_agent": "Finding hotels",
    "weather_agent": "Checking weather",
    "itinerary_agent": "Building itinerary",
}

AGENT_ICONS = {
    "intent_classifier": "🤔",
    "orchestrator": "🧠",
    "flight_agent": "✈️",
    "hotel_agent": "🏨",
    "weather_agent": "🌤️",
    "itinerary_agent": "🗺️",
}

# The orchestrator writes the final answer, so its tokens are the ones streamed.
STREAMED_ANSWER_NODE = "orchestrator"
RESULT_KEY_BY_AGENT = {spec.name: spec.result_key for spec in SUBAGENTS}


def _chunk_text(chunk) -> str:
    """Text of a streamed chat-model chunk (content may be a list of blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return content if isinstance(content, str) else ""


class AnswerStreamer:
    """Turns raw LLM tokens into guarded, line-sized `answer_delta` payloads.

    Tokens are buffered until a line is complete and each finished line goes
    through guard_output() before it leaves the server. Line granularity is
    what makes the guard meaningful mid-stream: the redaction patterns (API
    keys, card numbers) never span lines, whereas a single token can split
    one in half. The `complete` event still carries the fully guarded answer
    and replaces whatever was streamed.
    """

    def __init__(self) -> None:
        self._pending = ""
        self.started = False

    def feed(self, text: str) -> str | None:
        """Add tokens; return the newly completed lines (guarded), if any."""
        self._pending += text
        cut = self._pending.rfind("\n")
        if cut == -1:
            return None
        ready, self._pending = self._pending[: cut + 1], self._pending[cut + 1 :]
        return self._emit(ready)

    def flush(self) -> str | None:
        """Return whatever is left (the last line has no trailing newline)."""
        ready, self._pending = self._pending, ""
        return self._emit(ready) if ready else None

    def _emit(self, text: str) -> str:
        self.started = True
        return guard_output(text)


def _is_delegating(output) -> bool:
    """True when an orchestrator turn ended in tool calls rather than an answer."""
    messages = output.get("messages") if isinstance(output, dict) else None
    return bool(messages and getattr(messages[-1], "tool_calls", None))


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
      - `answer_delta`: the next line(s) of the final answer, as it is being written
      - `answer_reset`: discard the `answer_delta` text received so far — the
        orchestrator went on to delegate instead of answering
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
    final_state: dict | None = None
    started = time.perf_counter()
    answer = AnswerStreamer()

    try:
        async for event in graph.astream_events(plan.graph_input, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            in_answer_node = event.get("metadata", {}).get("langgraph_node") == STREAMED_ANSWER_NODE

            # Every orchestrator LLM call is a fresh candidate for the final
            # answer; only the last one counts. Drop what an earlier call streamed.
            if kind == "on_chat_model_start":
                if in_answer_node:
                    if answer.started:
                        yield _sse_event("answer_reset", {})
                    answer = AnswerStreamer()
                continue

            # Final answer tokens → line-buffered, guarded deltas.
            if kind == "on_chat_model_stream":
                if in_answer_node:
                    delta = answer.feed(_chunk_text(event.get("data", {}).get("chunk")))
                    if delta:
                        yield _sse_event("answer_delta", {"text": delta})
                continue

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

                output = event.get("data", {}).get("output", {})

                if name == STREAMED_ANSWER_NODE:
                    if _is_delegating(output):
                        # The turn ended in delegation — any text was not the answer.
                        if answer.started:
                            yield _sse_event("answer_reset", {})
                        answer = AnswerStreamer()
                    else:
                        tail = answer.flush()
                        if tail:
                            yield _sse_event("answer_delta", {"text": tail})

                yield _sse_event("progress", {
                    "agent": name,
                    "display": display_name,
                    "icon": icon,
                    "status": "completed",
                })

                # A sub-agent's report (preview) as soon as it is in
                result_key = RESULT_KEY_BY_AGENT.get(name)
                if result_key and isinstance(output, dict) and output.get(result_key):
                    yield _sse_event("agent_result", {result_key: guard_output(output[result_key][:500])})

            # Final graph completion
            if kind == "on_chain_end" and name == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_state = output
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
        log_run_summary(
            thread_id, final_state, started,
            outcome="awaiting_clarification" if clarification else "completed",
        )
        if clarification:
            yield _sse_event("clarification", {"thread_id": thread_id, **clarification})
        elif complete_payload is not None:
            yield _sse_event("complete", complete_payload)

    except Exception:
        logger.exception("Streaming graph execution failed for thread %s", thread_id)
        log_run_summary(thread_id, final_state, started, outcome="failed")
        yield _sse_event("error", {
            "message": "An error occurred while generating your travel plan. Please try again.",
            "thread_id": thread_id,
        })
