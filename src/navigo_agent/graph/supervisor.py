"""Supervisor node: LLM-driven routing for the multi-agent travel graph.

Since Groq's ChatGroq may not support native tool calling reliably,
the supervisor uses prompt-engineered JSON output with Pydantic parsing.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from navigo_agent.config import get_llm, MAX_SUPERVISOR_LOOPS
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

# ── Agent Name Normalization ──────────────────────────────────────────
# LLMs often output "flight" instead of "flight_agent", etc.
# This mapping prevents routing failures when the supervisor uses
# a short/alias name rather than the exact graph node key.

AGENT_NAME_ALIASES: dict[str, str] = {
    "flight": "flight_agent",
    "flight agent": "flight_agent",
    "flightagent": "flight_agent",
    "hotel": "hotel_agent",
    "hotel agent": "hotel_agent",
    "hotelagent": "hotel_agent",
    "weather": "weather_agent",
    "weather agent": "weather_agent",
    "weatheragent": "weather_agent",
    "itinerary": "itinerary_agent",
    "itinerary agent": "itinerary_agent",
    "itineraryagent": "itinerary_agent",
    "final": "final_synthesizer",
    "final_synthesizer": "final_synthesizer",
    "final synthesizer": "final_synthesizer",
    "finalsynthesizer": "final_synthesizer",
}

VALID_AGENT_NODES = frozenset({
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "itinerary_agent",
    "final_synthesizer",
})


def _normalize_agent_name(raw: str) -> str:
    """Map a raw agent name from the LLM to a valid graph node key.

    Handles common LLM output variations (case, spaces, suffixes).
    Falls back to exact match if the name is already valid.
    """
    cleaned = raw.strip().lower().replace("_", " ").replace("-", " ")
    # Collapse multiple spaces
    cleaned = " ".join(cleaned.split())

    # Direct alias lookup
    if cleaned in AGENT_NAME_ALIASES:
        return AGENT_NAME_ALIASES[cleaned]

    # Fuzzy: try appending "_agent"
    suffix_try = f"{cleaned.replace(' ', '_')}"
    if suffix_try in VALID_AGENT_NODES:
        return suffix_try

    # Already valid (with underscores)
    if raw.strip() in VALID_AGENT_NODES:
        return raw.strip()

    # Unknown — log and return as-is (caller should handle)
    logger.warning("Unrecognized agent name '%s' — passing through.", raw)
    return raw.strip()


# ── Routing Decision Schema ──────────────────────────────────────────

class SupervisorDecision(BaseModel):
    """Parsed routing decision from the supervisor LLM."""

    reasoning: str = Field(description="Why this routing decision was made")
    next_action: str = Field(
        description="One of: classify_intent, dispatch_flight, dispatch_hotel, "
        "dispatch_weather, dispatch_itinerary, synthesize_final, done"
    )
    agents_to_dispatch: list[str] = Field(
        default_factory=list,
        description="Agent names to run in parallel: flight_agent, hotel_agent, weather_agent",
    )
    is_complete: bool = Field(
        default=False,
        description="True when all needed information is gathered and ready for final synthesis",
    )


# ── Supervisor Prompt ────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent of a multi-agent travel planning system.

Your job: analyze the user's request and the current state of information gathering, then decide which specialized agents to dispatch next.

Available agents:
- flight_agent: searches flights via Tavily web search (ReAct multi-hop)
- hotel_agent: searches hotels via Tavily web search (ReAct multi-hop)
- weather_agent: fetches current weather + 5-day forecast via OpenWeatherMap
- itinerary_agent: generates a day-by-day itinerary from collected data (ReAct multi-hop)
- final_synthesizer: formats everything into a polished travel plan

Decision rules:
1. For weather-only queries ("weather in Tokyo"): dispatch ONLY weather_agent, then finalize
2. For flight-only queries ("flights to Paris"): dispatch ONLY flight_agent, then finalize
3. For hotel-only queries ("hotels in Dubai"): dispatch ONLY hotel_agent, then finalize
4. For full trip planning ("plan a 7-day Japan trip"): dispatch flight_agent + hotel_agent + weather_agent together, then itinerary_agent, then finalize
5. If flight/hotel/weather agents have run and returned results → dispatch itinerary_agent
6. When itinerary exists → set is_complete = true and next_action = "synthesize_final"
7. If an agent failed (check errors), skip it or retry once. Don't dispatch the same agent twice.

Output a JSON object with exactly these fields:
{
    "reasoning": "<one sentence explaining your decision>",
    "next_action": "<action from the list above>",
    "agents_to_dispatch": ["<agent_name>", ...] or [],
    "is_complete": true/false
}

IMPORTANT: Use the EXACT agent names: "flight_agent", "hotel_agent", "weather_agent", "itinerary_agent", "final_synthesizer".

Output ONLY the JSON. No markdown, no explanation outside the JSON.
"""


def _build_supervisor_prompt(state: TravelState) -> str:
    """Build the human message for the supervisor from current state."""
    completed = state.get("completed_agents", [])
    pending = state.get("pending_agents", [])
    errors = state.get("errors", [])
    loop_count = state.get("supervisor_loop_count", 0)

    has_flight = bool(state.get("flight_results"))
    has_hotel = bool(state.get("hotel_results"))
    has_weather = bool(state.get("weather_results"))
    has_itinerary = bool(state.get("itinerary"))

    return f"""Current State:
- User query: "{state.get('user_query', '')}"
- User intent: {state.get('user_intent', 'general')}
- Extracted destination: {state.get('extracted_destination', 'not yet extracted')}
- Supervisor loop: {loop_count}/{MAX_SUPERVISOR_LOOPS}
- Completed agents: {completed if completed else 'none'}
- Pending agents: {pending if pending else 'none'}
- Flight results: {'available' if has_flight else 'not yet'}
- Hotel results: {'available' if has_hotel else 'not yet'}
- Weather results: {'available' if has_weather else 'not yet'}
- Itinerary: {'available' if has_itinerary else 'not yet'}
- Errors: {errors if errors else 'none'}

Decide which agents to dispatch next."""


def _parse_supervisor_output(raw: str) -> SupervisorDecision:
    """Parse the supervisor LLM output into a SupervisorDecision.

    Handles malformed JSON by extracting the first JSON object found,
    and falls back to a safe default (dispatch all remaining, finalize).
    """
    # Try to extract JSON from the response (handles markdown code fences)
    text = raw.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        data = json.loads(text)
        # Normalize agent names from the LLM (e.g. "flight" → "flight_agent")
        raw_agents = data.get("agents_to_dispatch", [])
        if isinstance(raw_agents, list):
            data["agents_to_dispatch"] = [
                _normalize_agent_name(a) for a in raw_agents
            ]
        return SupervisorDecision(**data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse supervisor output: %s. Raw: %.200s", e, raw)
        # Fallback: signal completion to finalize with what we have
        return SupervisorDecision(
            reasoning="Fallback: could not parse routing decision, finalizing with available data.",
            next_action="synthesize_final",
            agents_to_dispatch=[],
            is_complete=True,
        )


# ── Supervisor Node ───────────────────────────────────────────────────

async def supervisor_node(state: TravelState) -> dict:
    """Supervisor agent node: evaluate state and decide routing.

    Called at graph entry (after intent classification) and after each
    batch of agents completes. Returns state updates with routing decisions.
    """
    loop_count = state.get("supervisor_loop_count", 0)

    # Safety bound: force finalization after MAX_SUPERVISOR_LOOPS
    if loop_count >= MAX_SUPERVISOR_LOOPS:
        logger.info("Supervisor loop limit reached (%d). Forcing finalization.", loop_count)
        return {
            "supervisor_reasoning": "Maximum supervisor loops reached. Finalizing.",
            "pending_agents": [],
            "completed_agents": state.get("completed_agents", []),
            "supervisor_loop_count": loop_count + 1,
        }

    # Build prompt and invoke LLM
    llm = get_llm(temperature=0.1)  # low temp for deterministic routing
    prompt = _build_supervisor_prompt(state)

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw_output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("Supervisor LLM call failed: %s", e)
        return {
            "supervisor_reasoning": f"LLM error: {e}. Finalizing with available data.",
            "pending_agents": [],
            "completed_agents": state.get("completed_agents", []),
            "supervisor_loop_count": loop_count + 1,
            "errors": [f"supervisor_llm_error: {str(e)}"],
        }

    decision = _parse_supervisor_output(raw_output)

    # Update tracking lists
    completed = list(state.get("completed_agents", []))
    pending = list(decision.agents_to_dispatch)

    logger.info(
        "Supervisor decision: action=%s agents=%s complete=%s reason=%s",
        decision.next_action,
        decision.agents_to_dispatch,
        decision.is_complete,
        decision.reasoning,
    )

    return {
        "supervisor_reasoning": decision.reasoning,
        "pending_agents": pending,
        "completed_agents": completed,
        "supervisor_loop_count": loop_count + 1,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
