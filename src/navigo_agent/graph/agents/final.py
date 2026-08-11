"""Final synthesizer: formats all collected data into a polished travel plan."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from navigo_agent.config import get_llm
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)


async def final_synthesizer(state: TravelState) -> dict:
    """Format all agent outputs into a polished, user-facing travel plan."""
    logger.info("Final synthesizer: formatting results")

    final_prompt = f"""
Generate a final travel plan based on the following information:
User Query: {state['user_query']}
Flight Results: {state.get('flight_results', 'No flight data available.')}
Hotel Results: {state.get('hotel_results', 'No hotel data available.')}
Weather Results: {state.get('weather_results', 'No weather data available.')}
Itinerary: {state.get('itinerary', 'No itinerary generated.')}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
- If some sections have no data, note that the information was unavailable rather than fabricating it.
"""

    try:
        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content="You are a professional travel assistant that creates a final travel plan based on user queries and search results."),
            HumanMessage(content=final_prompt),
        ])
    except Exception as e:  # noqa: BLE001
        logger.error("Final synthesizer failed: %s", e)
        return {
            "messages": [
                HumanMessage(
                    content="I apologize, but I encountered an error generating your travel plan. "
                    "Please try again or refine your request."
                )
            ],
            "llm_calls": state.get("llm_calls", 0),
            "errors": [f"final_synthesizer: {e!s}"],
        }

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
