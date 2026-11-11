"""Intent classification and conditional routing for the supervisor graph.

Intent classifier: first node — determines user intent and extracts trip slots.
Routing functions: conditional edges driven by supervisor decisions.
"""

import logging
import re

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from navigo_agent.graph.slots import (
    build_clarification_question,
    extract_trip_request,
    merge_slots,
    missing_required_slots,
)
from navigo_agent.graph.supervisor import VALID_AGENT_NODES, _normalize_agent_name
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

# ── Fast-Path Intent Patterns ────────────────────────────────────────

INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("weather", re.compile(
        r"\b(weather|temperature|forecast|rain|sunny|cloudy|hot|cold|humid|climate)\b", re.IGNORECASE
    )),
    ("flights", re.compile(
        r"\b(flights?|fly|flying|airlines?|airports?|depart(?:ure)?|arrive|arrival|airfare|one.way|round.trip)\b", re.IGNORECASE
    )),
    ("hotel", re.compile(
        r"\b(hotels?|motels?|hostels?|airbnb|accommodations?|lodging|stay|resorts?|inns?)\b", re.IGNORECASE
    )),
]

# Intents for which an itinerary is generated automatically once data is in
ITINERARY_INTENTS = frozenset({"full_trip", "itinerary"})

TRIP_PLANNING_KEYWORDS = [
    "plan", "trip", "travel", "visit", "vacation", "holiday",
    "itinerary", "tour", "days", "week", "budget",
]


def _fast_path_classify(query: str) -> str | None:
    """Try to classify intent via regex/keyword matching.

    Returns intent string or None if no clear match.
    """
    query_lower = query.lower()

    # Check for trip planning keywords first (most comprehensive intent)
    trip_score = sum(1 for kw in TRIP_PLANNING_KEYWORDS if kw in query_lower)
    if trip_score >= 2:
        return "full_trip"

    # Check specific intents
    scores: dict[str, int] = {}
    for intent, pattern in INTENT_PATTERNS:
        matches = len(pattern.findall(query_lower))
        if matches > 0:
            scores[intent] = matches

    if not scores:
        return None

    # Return highest-scoring intent
    best = max(scores, key=scores.get)

    # Single keyword match — return the intent (the keyword is distinctive enough)
    if scores[best] >= 1 and len(scores) == 1:
        return best

    # Multiple intent keywords matched — return the dominant one
    if scores[best] >= 2:
        return best

    return None  # ambiguous — let LLM classify


# ── Intent Classifier Node ───────────────────────────────────────────

async def intent_classifier(state: TravelState) -> dict:
    """Classify intent and extract structured trip slots in ONE LLM call.

    Intent: the regex fast path wins when it is confident; otherwise the
    extractor's intent is used. Slots (origin, destination, dates, budget...)
    always come from the extractor and are shared with every agent.
    """
    query = state["user_query"]

    fast_result = _fast_path_classify(query)
    llm_intent, slots = await extract_trip_request(query)

    # Extraction failed AND regex unsure → assume the most complete plan.
    intent = fast_result or llm_intent or "full_trip"
    logger.info(
        "Intent: %s (%s) | slots: %s",
        intent, "fast path" if fast_result else "LLM", slots.model_dump(exclude_none=True),
    )
    return {
        "user_intent": intent,
        "trip": slots.model_dump(),
        "extracted_destination": slots.destination,
        "supervisor_loop_count": 0,
        "llm_calls": 1,
    }


# ── Clarification Node (human-in-the-loop) ───────────────────────────

async def clarify_node(state: TravelState) -> dict:
    """Ask the user for essential details that are missing — at most once.

    Uses LangGraph's `interrupt()`: the run pauses here (state is saved by the
    checkpointer) and the question is returned to the caller. When the caller
    resumes the thread with the user's reply, this node re-runs from the top
    and `interrupt()` returns that reply.

    There is no second round: after one reply the graph proceeds with whatever
    is known, so an "I don't know" answer can never trap the user in a loop.
    """
    if not state.get("allow_clarification", True):
        return {}

    missing = missing_required_slots(state.get("user_intent", ""), state.get("trip"))
    if not missing:
        return {}

    logger.info("Clarification needed — missing slots: %s", missing)
    reply = interrupt({
        "question": build_clarification_question(missing),
        "missing": missing,
    })

    # ── Resumed with the user's reply ────────────────────────────────
    reply_text = str(reply or "").strip()
    if not reply_text:
        return {}

    combined_query = f"{state['user_query']}\n\nAdditional details from the user: {reply_text}"
    _, new_slots = await extract_trip_request(combined_query)
    trip = merge_slots(state.get("trip"), new_slots)
    logger.info("Clarification received. Slots now: %s", trip)
    return {
        "user_query": combined_query,  # agents see the extra details too
        "messages": [HumanMessage(content=reply_text)],
        "trip": trip,
        "extracted_destination": trip.get("destination"),
        "llm_calls": 1,
    }


# ── Routing Functions (for conditional edges) ────────────────────────

# Agents that gather independent data and can safely run side by side.
PARALLEL_AGENTS = ("flight_agent", "hotel_agent", "weather_agent")


def _valid_pending(pending: list[str]) -> list[str]:
    """Normalize + validate the supervisor's dispatch list (order kept, no dupes)."""
    valid: list[str] = []
    for agent in pending:
        normalized = _normalize_agent_name(agent)
        if normalized not in VALID_AGENT_NODES:
            logger.warning("Skipping invalid agent name '%s' — not a valid graph node.", agent)
        elif normalized not in valid:
            valid.append(normalized)
    return valid


def route_after_supervisor(state: TravelState) -> str | list[str]:
    """Fan out to the supervisor's dispatch batch.

    Returns a LIST of node names when several data agents are dispatched —
    LangGraph runs them concurrently in one superstep and the graph only
    returns to the supervisor once all of them have finished.

    itinerary_agent consumes the data agents' output, so it never runs in the
    same batch as them: it is deferred until their results are in.
    """
    pending = _valid_pending(state.get("pending_agents", []))

    data_agents = [a for a in pending if a in PARALLEL_AGENTS]
    if data_agents:
        logger.info("Dispatching in parallel: %s", data_agents)
        return data_agents

    if "itinerary_agent" in pending:
        return "itinerary_agent"

    # Nothing (valid) dispatched. Only trip-planning intents get an automatic
    # itinerary; single-topic queries (weather / flights / hotel) go straight
    # to the final answer unless the supervisor explicitly dispatched it.
    if state.get("user_intent") not in ITINERARY_INTENTS:
        return "final_synthesizer"

    has_data = bool(state.get("flight_results") or state.get("hotel_results") or state.get("weather_results"))
    has_itinerary = bool(state.get("itinerary"))
    completed = state.get("completed_agents", [])

    if has_data and not has_itinerary and "itinerary_agent" not in completed:
        return "itinerary_agent"

    return "final_synthesizer"
