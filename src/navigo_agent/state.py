"""TravelState TypedDict for the supervisor-driven LangGraph.

Extended from the original linear pipeline to support:
  - Intent classification + structured trip slots
  - Supervisor routing decisions
  - Parallel agent dispatch (fan-out) with reducer-merged tracking fields
  - Error accumulation (graceful degradation)

Reducers matter here: flight / hotel / weather agents run in the SAME graph
superstep, so any key more than one of them writes must declare how the
concurrent writes are merged. Such nodes return *deltas* (their own calls,
their own errors, their own name), never a rebuilt copy of the old value.
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
    """State flowing through the supervisor-orchestrated travel graph."""

    # ── Core ──────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # ── Intent & Routing ──────────────────────────────────────────
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
    supervisor_reasoning: str  # explainability: why this routing decision
    supervisor_loop_count: int  # safety bound, max 3

    # ── Agent Results ─────────────────────────────────────────────
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # ── Agent Tracking ────────────────────────────────────────────
    pending_agents: list[str]  # current dispatch batch — written ONLY by the supervisor
    completed_agents: Annotated[list[str], merge_unique]  # agents append their own name

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
