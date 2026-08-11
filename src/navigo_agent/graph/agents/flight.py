"""Flight agent: ReAct loop over Tavily web search.

Uses the shared react_utils module for the ReAct loop — only provides
the system prompt builder and the search function (Tavily via MCP).
"""

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from navigo_agent.config import get_llm
from navigo_agent.state import TravelState
from navigo_agent.mcp import tavily_mcp_search, extract_mcp_text
from navigo_agent.graph.agents.react_utils import run_react_loop

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 3


# ── Prompt builder ────────────────────────────────────────────────────

def _build_react_system_prompt(destination: str, user_query: str) -> str:
    """Build the ReAct system prompt with output format and search guidance."""
    return f"""You are an expert flight travel researcher. You have access to a Tavily web search tool that returns real-time flight, airline, and travel information.

## Your Goal

Answer the user's flight-related query thoroughly. To do this, you can run up to {MAX_REACT_ITERATIONS} web searches before producing your final answer.

## Search Strategy (Multi-Hop)

Don't try to get everything in one search. Instead:
1. **First search**: broad flight info — "{destination} flights airlines routes airports"
2. **Review gaps**: do you have airline names? price ranges? airport codes?
3. **Second search** (if needed): target the gap — e.g. "Singapore Airlines Tokyo Narita direct flight price 2026"
4. **Third search** (if needed): practical tips — baggage, booking timing, peak season for the route

## Output Format

Respond with ONLY a JSON object (no markdown, no extra text):

**To search the web:**
{{"reasoning": "I have general route info but no specific airline names. I need to search for airlines on this route.", "action": "search", "query": "airlines flying Singapore to Tokyo direct flights 2026"}}

**To give your final answer:**
{{"reasoning": "I now have airport codes (SIN → NRT/HND), major airlines (Singapore Airlines, ANA, JAL, Scoot, Zipair), typical prices, and booking tips.", "action": "final_answer", "content": "## Flight Options: Singapore → Tokyo\\n\\n**Airports:** ...\\n\\n**Airlines:** ...\\n\\n**Estimated Airfare:** ...\\n\\n**Tips:** ..."}}

## Final Answer Requirements

Cover these points (mark "N/A — not found in search" if genuinely unavailable):
1. Departure & arrival airports with IATA codes
2. Airlines operating the route (full-service + budget)
3. Typical flight duration
4. Estimated economy round-trip airfare range
5. Best booking window and peak season warning
6. Practical tips: check-in baggage policies, stopover options

## Rules

- Each search query should be specific and different from previous ones — refine, don't repeat
- If a search returns nothing useful, try a broader/different angle
- Never fabricate specific prices or flight numbers
- After {MAX_REACT_ITERATIONS} searches, you MUST produce a final answer with whatever you have

## User's Travel Query

{user_query}"""


# ── Search function (Tavily wrapper) ──────────────────────────────────

async def _search_flights(query: str) -> str:
    """Call Tavily MCP and unwrap result."""
    raw = await tavily_mcp_search(query)
    return extract_mcp_text(raw)


# ── Flight Agent Node ─────────────────────────────────────────────────

async def flight_agent(state: TravelState) -> dict:
    """ReAct loop: reason → search Tavily → observe → repeat → final answer.

    Delegates to the shared run_react_loop(), providing only domain-specific
    prompt building and the Tavily search function.
    """
    query = state["user_query"]
    destination = state.get("extracted_destination") or query
    logger.info("Flight agent (ReAct): processing '%s'", query[:80])

    system_prompt = _build_react_system_prompt(destination, query)
    conversation: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Find flight information for: {query}"),
    ]

    llm = get_llm(temperature=0.2)

    final_answer, llm_calls, errors = await run_react_loop(
        llm=llm,
        conversation=conversation,
        search_fn=_search_flights,
        agent_name="flight_agent",
        max_iterations=MAX_REACT_ITERATIONS,
        llm_calls=state.get("llm_calls", 0),
    )

    # ── Build state update ─────────────────────────────────────────────
    completed = list(set(state.get("completed_agents", []) + ["flight_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "flight_agent"]

    return {
        "flight_results": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "llm_calls": llm_calls,
        "errors": state.get("errors", []) + errors,
        "completed_agents": completed,
        "pending_agents": pending,
    }
