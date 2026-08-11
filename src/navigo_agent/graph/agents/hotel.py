"""Hotel agent: ReAct loop over Tavily web search.

Uses the shared react_utils module — only provides the system prompt
builder and the Tavily search function.
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
    """Build the ReAct system prompt with hotel search strategy."""
    return f"""You are an expert hotel and accommodation researcher. You have access to a Tavily web search tool that returns real-time hotel, neighbourhood, and travel accommodation information.

## Your Goal

Answer the user's hotel-related query thoroughly. To do this, you can run up to {MAX_REACT_ITERATIONS} web searches before producing your final answer.

## Search Strategy (Multi-Hop)

Don't try to get everything in one search. Instead:
1. **First search**: broad hotel landscape — "best hotels and areas to stay in {destination}"
2. **Review gaps**: do you have specific neighbourhoods? price ranges? hotel names? standout amenities?
3. **Second search** (if needed): target the gap — e.g. "luxury hotels Shinjuku Tokyo 2026" or "budget hostels Asakusa Tokyo"
4. **Third search** (if needed): practical tips — best booking site, seasonal pricing, cancellation policies for {destination}

## Output Format

Respond with ONLY a JSON object (no markdown, no extra text):

**To search the web:**
{{"reasoning": "I have general area recommendations but no specific hotel names or price ranges in Shibuya. I need to search for hotels in that neighbourhood.", "action": "search", "query": "best hotels Shibuya Tokyo mid-range 2026"}}

**To give your final answer:**
{{"reasoning": "I now have neighbourhood breakdown (Shinjuku, Shibuya, Asakusa, Ginza), specific hotel names with price ranges, and booking tips.", "action": "final_answer", "content": "## Where to Stay in Tokyo\\n\\n**Shinjuku** — Best for first-time visitors...\\n\\n### Top Picks\\n\\n| Hotel | Area | Price | Best For |\\n|-------|------|-------|----------|\\n| ... | ... | ... | ... |\\n\\n### Booking Tips\\n..."}}

## Final Answer Requirements

Cover these points (mark "N/A — not found in search" if genuinely unavailable):
1. Best neighbourhoods/areas to stay with a brief description of each
2. 4-6 specific hotel recommendations covering different budgets (budget, mid-range, luxury)
3. For each hotel: name, general location, star rating or price range, key amenities, best for (couples/families/solo/business)
4. Best booking window and peak/low season pricing notes
5. Practical tips: best booking platforms, cancellation policies, local accommodation quirks

## Rules

- Each search query should be specific and different from previous ones — refine, don't repeat
- If a search returns nothing useful, try a broader/different angle
- Never fabricate specific prices or hotel names — say "N/A — not found in search" if unavailable
- After {MAX_REACT_ITERATIONS} searches, you MUST produce a final answer with whatever you have
- Use markdown formatting (headings, tables, bullet points) for readability

## User's Travel Query

{user_query}"""


# ── Search function (Tavily wrapper) ──────────────────────────────────

async def _search_hotels(query: str) -> str:
    """Call Tavily MCP and unwrap result."""
    raw = await tavily_mcp_search(query)
    return extract_mcp_text(raw)


# ── Hotel Agent Node ──────────────────────────────────────────────────

async def hotel_agent(state: TravelState) -> dict:
    """ReAct loop: reason → search Tavily → observe → repeat → final answer."""
    query = state["user_query"]
    destination = state.get("extracted_destination") or query
    logger.info("Hotel agent (ReAct): processing '%s'", query[:80])

    system_prompt = _build_react_system_prompt(destination, query)
    conversation: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Find hotel and accommodation information for: {query}"),
    ]

    llm = get_llm(temperature=0.3)

    final_answer, llm_calls, errors = await run_react_loop(
        llm=llm,
        conversation=conversation,
        search_fn=_search_hotels,
        agent_name="hotel_agent",
        max_iterations=MAX_REACT_ITERATIONS,
        llm_calls=state.get("llm_calls", 0),
    )

    # ── Build state update ─────────────────────────────────────────────
    completed = list(set(state.get("completed_agents", []) + ["hotel_agent"]))
    pending = [a for a in state.get("pending_agents", []) if a != "hotel_agent"]

    return {
        "hotel_results": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "llm_calls": llm_calls,
        "errors": state.get("errors", []) + errors,
        "completed_agents": completed,
        "pending_agents": pending,
    }
