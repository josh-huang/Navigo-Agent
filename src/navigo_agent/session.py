"""Run/session helpers shared by the JSON and SSE entry points.

A request is one of:
  - a NEW run     → fresh thread, fresh initial state
  - a RESUME      → the client's thread is paused on a clarification question
                    (LangGraph interrupt) and this message is the user's reply
  - a FOLLOW-UP   → the client's thread is a finished turn; a fresh thread is
                    started, seeded with that turn's history and trip slots

Finished threads are never re-run: reducer-backed state (messages, errors,
llm_calls, completed_agents) must not leak from one request into the next.
Only the conversation history and slots are carried forward, explicitly.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from navigo_agent.graph.slots import append_turn
from navigo_agent.state import build_initial_state

logger = logging.getLogger(__name__)


@dataclass
class RunPlan:
    graph_input: Any  # initial state dict, or Command(resume=...)
    config: dict
    thread_id: str
    resumed: bool


def _interrupt_payload(interrupts) -> dict | None:
    """Extract our {"question", "missing"} payload from LangGraph Interrupt objects."""
    for item in interrupts or ():
        value = getattr(item, "value", item)
        if isinstance(value, dict) and value.get("question"):
            return {"question": str(value["question"]), "missing": list(value.get("missing") or [])}
    return None


def _pending_from_snapshot(snapshot) -> dict | None:
    if not getattr(snapshot, "next", None):
        return None
    interrupts = getattr(snapshot, "interrupts", None)
    if not interrupts:
        interrupts = [i for task in getattr(snapshot, "tasks", ()) for i in getattr(task, "interrupts", ())]
    return _interrupt_payload(interrupts)


async def pending_clarification(graph, config: dict) -> dict | None:
    """Return the pending clarification payload for a thread, or None."""
    return _pending_from_snapshot(await graph.aget_state(config))


def clarification_from_result(result: dict) -> dict | None:
    """Clarification payload from an ainvoke() result (`__interrupt__` key), or None."""
    if not isinstance(result, dict):
        return None
    return _interrupt_payload(result.get("__interrupt__"))


def _previous_turn(snapshot) -> tuple[list[dict[str, str]], dict] | None:
    """(history incl. the finished turn, its slots) from a completed thread, or None."""
    if getattr(snapshot, "next", None):
        return None  # still running / paused — not a finished turn
    values = getattr(snapshot, "values", None) or {}
    query = values.get("user_query")
    messages = values.get("messages") or []
    if not query or not messages:
        return None
    answer = getattr(messages[-1], "content", "")
    if not isinstance(answer, str) or not answer.strip():
        return None
    return append_turn(values.get("history"), str(query), answer), dict(values.get("trip") or {})


async def plan_run(
    graph,
    user_input: str,
    thread_id: str | None,
    *,
    allow_clarification: bool = True,
) -> RunPlan:
    """Decide how this message relates to the client's `thread_id`.

    - thread paused on a clarification → RESUME it with this message as the reply
    - thread finished                   → new run, seeded as a FOLLOW-UP of that turn
    - no / unknown thread               → new, independent run

    Every run except a resume gets a fresh thread id.
    """
    history: list[dict[str, str]] = []
    previous_trip: dict = {}

    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await graph.aget_state(config)
            pending = _pending_from_snapshot(snapshot)
            previous = None if pending else _previous_turn(snapshot)
        except Exception:
            logger.exception("Could not inspect thread %s — starting a new run.", thread_id)
            pending, previous = None, None
        if pending:
            logger.info("Resuming thread %s with the user's clarification.", thread_id)
            return RunPlan(Command(resume=user_input), config, thread_id, resumed=True)
        if previous:
            history, previous_trip = previous
            logger.info("Follow-up to thread %s (%d earlier turn(s)).", thread_id, len(history))

    new_thread_id = f"user_{uuid.uuid4().hex}"
    return RunPlan(
        build_initial_state(
            user_input,
            allow_clarification=allow_clarification,
            history=history,
            previous_trip=previous_trip,
        ),
        {"configurable": {"thread_id": new_thread_id}},
        new_thread_id,
        resumed=False,
    )


def log_run_summary(thread_id: str, state: dict | None, started: float, *, outcome: str) -> None:
    """One line per request: what was asked for, what it cost, what went wrong.

    `started` is a time.perf_counter() value. With LOG_FORMAT=json the extras
    become fields, so runs can be filtered by intent, duration or error count.
    """
    state = state if isinstance(state, dict) else {}
    errors = list(state.get("errors") or [])
    duration_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "Run %s: thread=%s intent=%s agents=%s llm_calls=%s errors=%d duration_ms=%d",
        outcome, thread_id, state.get("user_intent") or "-",
        ",".join(state.get("completed_agents") or []) or "-",
        state.get("llm_calls", 0), len(errors), duration_ms,
        extra={
            "thread_id": thread_id,
            "outcome": outcome,
            "intent": state.get("user_intent"),
            "agents": list(state.get("completed_agents") or []),
            "llm_calls": state.get("llm_calls", 0),
            "errors": errors,
            "duration_ms": duration_ms,
        },
    )
