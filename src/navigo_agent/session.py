"""Run/session helpers shared by the JSON and SSE entry points.

A request is either:
  - a NEW run     → fresh thread, fresh initial state, or
  - a RESUME      → the client's thread is paused on a clarification question
                    (LangGraph interrupt) and this message is the user's reply.

Finished threads are never reused: reducer-backed state (messages, errors,
llm_calls, completed_agents) must not leak from one request into the next.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

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


async def pending_clarification(graph, config: dict) -> dict | None:
    """Return the pending clarification payload for a thread, or None."""
    snapshot = await graph.aget_state(config)
    if not getattr(snapshot, "next", None):
        return None
    interrupts = getattr(snapshot, "interrupts", None)
    if not interrupts:
        interrupts = [i for task in getattr(snapshot, "tasks", ()) for i in getattr(task, "interrupts", ())]
    return _interrupt_payload(interrupts)


def clarification_from_result(result: dict) -> dict | None:
    """Clarification payload from an ainvoke() result (`__interrupt__` key), or None."""
    if not isinstance(result, dict):
        return None
    return _interrupt_payload(result.get("__interrupt__"))


async def plan_run(
    graph,
    user_input: str,
    thread_id: str | None,
    *,
    allow_clarification: bool = True,
) -> RunPlan:
    """Decide whether this message resumes a paused thread or starts a new run."""
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            pending = await pending_clarification(graph, config)
        except Exception:
            logger.exception("Could not inspect thread %s — starting a new run.", thread_id)
            pending = None
        if pending:
            logger.info("Resuming thread %s with the user's clarification.", thread_id)
            return RunPlan(Command(resume=user_input), config, thread_id, resumed=True)

    new_thread_id = f"user_{uuid.uuid4().hex}"
    return RunPlan(
        build_initial_state(user_input, allow_clarification=allow_clarification),
        {"configurable": {"thread_id": new_thread_id}},
        new_thread_id,
        resumed=False,
    )
