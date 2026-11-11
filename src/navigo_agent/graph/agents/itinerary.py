"""Itinerary agent: ReAct loop over Tavily web search (native tool calling).

Uses the shared react_utils module. Unlike flight/hotel, this agent
starts with pre-collected data (flight, hotel, weather) already in state
and searches for attraction/transport/dining gaps.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from navigo_agent.config import get_llm
from navigo_agent.graph.agents.react_utils import run_react_loop
from navigo_agent.graph.slots import conversation_block, format_trip_details
from navigo_agent.mcp import extract_mcp_text, tavily_mcp_search
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 3


# ── Prompt builder ────────────────────────────────────────────────────

def _build_react_system_prompt(
    destination: str,
    user_query: str,
    flight_data: str,
    hotel_data: str,
    weather_data: str,
    trip_details: str,
) -> str:
    """Build the ReAct system prompt — includes pre-collected peer agent data."""
    return f"""You are an expert travel itinerary planner. You have access to a `web_search` tool (Tavily) for real-time attraction, transport, and local information.

## Pre-Collected Data

Your peer agents have already gathered the following data. Use it — don't re-search for flights, hotels, or weather.

### Flight Information
{flight_data or "No flight data available."}

### Hotel Information
{hotel_data or "No hotel data available."}

### Weather Information
{weather_data or "No weather data available."}

## Your Goal

Create a detailed, practical day-by-day travel itinerary for the user. You can run up to {MAX_REACT_ITERATIONS} rounds of web searches to fill gaps before producing your final plan.

## Search Strategy (Multi-Hop)

Don't try to get everything in one search. Instead:
1. **First search**: top attractions and must-see sights — "{destination} top attractions must-see sights 2026"
2. **Review gaps**: do you have enough for each day? Missing transport logistics? Restaurant recommendations? Opening hours? Activity sequencing?
3. **Second search** (if needed): target the gap — e.g. "{destination} to Kyoto train schedule price" or "{destination} Shinjuku to Asakusa travel time"
4. **Third search** (if needed): local tips — "{destination} hidden gems local food neighbourhood guide" or seasonal events

## How to Work

- To look something up, call the `web_search` tool with a specific query. You may call it several times.
- When you have enough information, reply with your final answer as plain markdown — no JSON, no tool call.
- Structure the final answer as `### Day N: <theme>` sections with Morning / Lunch / Afternoon / Evening blocks.

## Final Answer Requirements

1. Day-by-day breakdown with morning / afternoon / evening blocks
2. For each activity: brief description, estimated duration, travel time from previous location
3. Lunch and dinner recommendations near the day's activities
4. Weather-aware suggestions (indoor alternatives for rainy days, seasonal tips)
5. Budget notes: which activities are free vs ticketed, rough cost estimates
6. Practical tips: transport passes, advance booking needed, local customs

## Rules

- Each search query should be specific and different from previous ones — refine, don't repeat
- If a search returns nothing useful, try a broader/different angle
- Never fabricate opening hours, ticket prices, or train schedules — say "check official website"
- Leverage the pre-collected flight/hotel/weather data — don't re-search those topics
- After {MAX_REACT_ITERATIONS} searches, you MUST produce a final itinerary with whatever you have
- Use markdown formatting (## headings, bullet points, **bold** for times/locations) for readability

## Trip Details (extracted from the query)

{trip_details}

## User's Travel Query

{user_query}"""


# ── Search function (Tavily wrapper) ──────────────────────────────────

async def _search_itinerary(query: str) -> str:
    """Call Tavily MCP and unwrap result."""
    raw = await tavily_mcp_search(query)
    return extract_mcp_text(raw)


# ── Itinerary Agent Node ──────────────────────────────────────────────

async def itinerary_agent(state: TravelState) -> dict:
    """ReAct loop: reason → search Tavily → observe → repeat → final itinerary.

    Starts with pre-collected flight/hotel/weather data from state.
    The ReAct loop fills gaps: attractions, transport logistics, dining.
    """
    query = state["user_query"]
    destination = state.get("extracted_destination") or query
    flight_data = state.get("flight_results", "")
    hotel_data = state.get("hotel_results", "")
    weather_data = state.get("weather_results", "")
    logger.info("Itinerary agent (ReAct): processing '%s'", query[:80])

    system_prompt = _build_react_system_prompt(
        destination, query, flight_data, hotel_data, weather_data,
        format_trip_details(state.get("trip")),
    )
    conversation: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create a detailed day-by-day travel itinerary for: {query}{conversation_block(state.get('history'))}"),
    ]

    llm = get_llm(temperature=0.5)  # higher temp for creative planning variety

    final_answer, llm_calls, errors = await run_react_loop(
        llm=llm,
        conversation=conversation,
        search_fn=_search_itinerary,
        agent_name="itinerary_agent",
        max_iterations=MAX_REACT_ITERATIONS,
    )

    # State update: deltas only — reducers merge them with parallel agents' writes.
    return {
        "itinerary": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "llm_calls": llm_calls,
        "errors": errors,
        "completed_agents": ["itinerary_agent"],
    }
