"""TravelState TypedDict for the supervisor-driven LangGraph.

Extended from the original linear pipeline to support:
  - Intent classification
  - Supervisor routing decisions
  - Parallel agent dispatch tracking
  - Error accumulation (graceful degradation)
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage


class TravelState(TypedDict):
    """State flowing through the supervisor-orchestrated travel graph."""

    # ── Core ──────────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # ── Intent & Routing ──────────────────────────────────────────
    user_intent: str  # flight | hotel | weather | itinerary | full_trip | general
    extracted_destination: str | None  # parsed early, reused by weather + flight
    supervisor_reasoning: str  # explainability: why this routing decision
    supervisor_loop_count: int  # safety bound, max 3

    # ── Agent Results ─────────────────────────────────────────────
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # ── Agent Tracking ────────────────────────────────────────────
    pending_agents: list[str]  # agents dispatched but not yet completed
    completed_agents: list[str]  # agents that have finished

    # ── Metrics & Errors ──────────────────────────────────────────
    llm_calls: int
    errors: Annotated[list[str], operator.add]
