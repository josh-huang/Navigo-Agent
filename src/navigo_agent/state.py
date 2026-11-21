"""TravelState TypedDict for the orchestrator-driven LangGraph.

One main agent (the orchestrator) owns the request: it delegates research to
specialist sub-agents through tool calls, reads their reports and writes the
final answer. The state carries:
  - Intent classification + structured trip slots
  - The orchestrator's transcript (`messages`): its delegation tool calls and
    the sub-agents' reports as ToolMessages, ending with the final answer
  - Per-sub-agent results, tracking and error accumulation

Reducers matter here: sub-agents delegated in one round run in the SAME graph
superstep, so any key more than one of them writes must declare how the
concurrent writes are merged. Such nodes return *deltas* (their own report,
their own calls, their own errors, their own name), never a rebuilt copy of
the old value.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage


def merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer: order-preserving union (used for completed_agents)."""
    merged = list(left or [])
    for item in right or []:
        if item not in merged:
            merged.append(item)
    return merged


class TravelState(TypedDict):
    """State flowing through the orchestrator-driven travel graph."""

    # ── Core ──────────────────────────────────────────────────────
    # The user's message(s), then the orchestrator's transcript:
    #   AIMessage(tool_calls=…) → one ToolMessage per delegated sub-agent → …
    # The last message of a finished run is the final answer.
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # ── Intent & Slots ────────────────────────────────────────────
    user_intent: str  # flight | hotel | weather | itinerary | full_trip | general
    trip: dict[str, Any]  # structured slots (see graph/slots.py: TripSlots)
    # Multi-turn: earlier turns of this conversation, oldest first:
    #   [{"user": "<request>", "assistant": "<final answer, truncated>"}]
    # Seeded from the previous turn's thread; emptied if this message starts a new topic.
    history: list[dict[str, str]]
    previous_trip: dict[str, Any]  # slots of the previous turn (carried into follow-ups)
    extracted_destination: str | None  # == trip["destination"], kept for convenience
    allow_clarification: bool  # False → never pause to ask the user (batch / eval runs)
    clarification_question: str  # the question to ask, already in the user's language

    # ── Orchestration ─────────────────────────────────────────────
    orchestrator_rounds: int  # delegation rounds issued so far — written ONLY by the orchestrator
    # Present only in the payload a sub-agent receives via Send():
    #   {"id": "<tool call id>", "args": {...}}
    # Sub-agents never write it back, so it stays empty in the graph state.
    delegation: dict[str, Any]

    # ── Sub-agent Results ─────────────────────────────────────────
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # ── Sub-agent Tracking ────────────────────────────────────────
    completed_agents: Annotated[list[str], merge_unique]  # sub-agents append their own name

    # ── Metrics & Errors ──────────────────────────────────────────
    llm_calls: Annotated[int, operator.add]  # nodes return the number of calls THEY made
    errors: Annotated[list[str], operator.add]


def build_initial_state(
    user_input: str,
    *,
    allow_clarification: bool = True,
    history: list[dict[str, str]] | None = None,
    previous_trip: dict | None = None,
) -> dict:
    """Fresh input state for one graph run (shared by invoke + streaming paths)."""
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "user_intent": "",
        "trip": {},
        "history": list(history or []),
        "previous_trip": dict(previous_trip or {}),
        "extracted_destination": None,
        "allow_clarification": allow_clarification,
        "clarification_question": "",
        "orchestrator_rounds": 0,
        "delegation": {},
        "flight_results": "",
        "hotel_results": "",
        "weather_results": "",
        "itinerary": "",
        "completed_agents": [],
        "llm_calls": 0,
        "errors": [],
    }
