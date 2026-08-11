"""Intent classification and conditional routing for the supervisor graph.

Intent classifier: lightweight first node that determines user intent.
Routing functions: conditional edges driven by supervisor decisions.
"""

import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from navigo_agent.config import get_llm
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

# ── Fast-Path Intent Patterns ────────────────────────────────────────

INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("weather", re.compile(
        r"\b(weather|temperature|forecast|rain|sunny|cloudy|hot|cold|humid|climate)\b", re.I
    )),
    ("flights", re.compile(
        r"\b(flights?|fly|flying|airlines?|airports?|depart(?:ure)?|arrive|arrival|airfare|one.way|round.trip)\b", re.I
    )),
    ("hotel", re.compile(
        r"\b(hotels?|motels?|hostels?|airbnb|accommodations?|lodging|stay|resorts?|inns?)\b", re.I
    )),
]

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


async def _llm_classify(query: str) -> str:
    """Use LLM to classify intent when fast path can't determine it."""
    llm = get_llm(temperature=0.1)

    prompt = f"""Classify this travel-related query into exactly one category.

Categories:
- flight: only asking about flights, airlines, airports
- hotel: only asking about hotels, accommodations, places to stay
- weather: only asking about weather, temperature, climate
- itinerary: asking for a day-by-day plan based on already-known information
- full_trip: asking for a complete trip plan including flights, hotels, and activities
- general: none of the above, but still travel-related

Query: "{query}"

Respond with ONLY the category name (one word). No explanation."""

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You classify travel queries into intent categories."),
            HumanMessage(content=prompt),
        ])
        result = response.content.strip().lower() if hasattr(response, "content") else "general"
        valid = {"flight", "hotel", "weather", "itinerary", "full_trip", "general"}
        return result if result in valid else "general"
    except Exception as e:
        logger.warning("Intent classification LLM call failed: %s. Defaulting to full_trip.", e)
        return "full_trip"


# ── Intent Classifier Node ───────────────────────────────────────────

async def intent_classifier(state: TravelState) -> dict:
    """Classify user intent: fast-path regex first, LLM fallback if ambiguous."""
    query = state["user_query"]

    # Fast path
    fast_result = _fast_path_classify(query)
    if fast_result:
        logger.info("Intent classified (fast path): %s → %s", query[:80], fast_result)
        return {
            "user_intent": fast_result,
            "supervisor_loop_count": 0,
            "completed_agents": [],
        }

    # LLM fallback
    llm_result = await _llm_classify(query)
    logger.info("Intent classified (LLM): %s → %s", query[:80], llm_result)
    return {
        "user_intent": llm_result,
        "supervisor_loop_count": 0,
        "completed_agents": [],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ── Routing Functions (for conditional edges) ────────────────────────

def route_after_intent(state: TravelState) -> str:
    """After intent classification, always route to supervisor."""
    return "supervisor"


def route_after_supervisor(state: TravelState) -> str:
    """Route based on supervisor's pending_agents list.

    Returns next agent node to visit, or falls through to final.
    """
    pending = state.get("pending_agents", [])

    if pending:
        next_agent = pending[0]
        logger.info("Routing to next pending agent: %s", next_agent)
        return next_agent

    # No pending agents — check if itinerary should run
    has_data = bool(state.get("flight_results") or state.get("hotel_results") or state.get("weather_results"))
    has_itinerary = bool(state.get("itinerary"))
    completed = state.get("completed_agents", [])

    if has_data and not has_itinerary and "itinerary_agent" not in completed:
        return "itinerary_agent"

    return "final_synthesizer"


def route_after_agent(state: TravelState) -> str:
    """After an individual agent completes, decide what's next.

    Check if more pending agents exist; if not, loop back to supervisor
    for re-evaluation.
    """
    pending = state.get("pending_agents", [])
    loop_count = state.get("supervisor_loop_count", 0)
    from navigo_agent.config import MAX_SUPERVISOR_LOOPS

    # Move just-completed agent from pending to completed
    # (the agent node itself handles this; here we just check remaining)

    if pending:
        next_agent = pending[0]
        logger.info("Dispatching next pending agent: %s (remaining: %s)", next_agent, pending[1:])
        return next_agent

    # All dispatched agents done → back to supervisor for re-evaluation
    if loop_count < MAX_SUPERVISOR_LOOPS:
        return "supervisor"

    return "final_synthesizer"
