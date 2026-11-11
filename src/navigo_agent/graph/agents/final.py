"""Final synthesizer: formats all collected data into a polished travel plan."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from navigo_agent.config import get_llm
from navigo_agent.graph.slots import (
    conversation_block,
    format_trip_details,
    reply_language_instruction,
)
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)


TRIP_INTENTS = frozenset({"full_trip", "itinerary"})

FULL_PLAN_INSTRUCTIONS = """Format the final answer beautifully using these sections:

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
- If some sections have no data, note that the information was unavailable rather than fabricating it."""

FOCUSED_INSTRUCTIONS = """Answer the user's question directly, using ONLY the information above.

Important:
- Cover only what the user asked about. Do NOT add sections for topics they did not
  ask about (no flight, hotel, itinerary or budget sections unless requested).
- Lead with the answer, then the supporting details. Use markdown headings and
  bullet points where they help; keep it concise.
- If the information needed is missing above, say it was unavailable rather than fabricating it."""


def build_final_prompt(state: TravelState) -> str:
    """Build the synthesis prompt: only the data that exists, in a shape that fits the intent.

    Trip-planning intents get the full 7-section plan. Single-topic questions
    (weather / flights / hotel / general) get a focused answer instead of a
    plan template padded with "information unavailable" sections.
    """
    data_blocks = [
        ("Flight Results", state.get("flight_results")),
        ("Hotel Results", state.get("hotel_results")),
        ("Weather Results", state.get("weather_results")),
        ("Itinerary", state.get("itinerary")),
    ]
    is_trip = state.get("user_intent") in TRIP_INTENTS
    if is_trip:
        # The plan template names every section, so say explicitly what is missing.
        data = "\n".join(f"{label}: {value or 'No data available.'}" for label, value in data_blocks)
    else:
        data = "\n".join(f"{label}: {value}" for label, value in data_blocks if value)
        data = data or "No data was gathered for this request."

    # Follow-up: show the previous answer at length so it is revised, not restarted.
    follow_up = conversation_block(state.get("history"), answer_chars=1500)
    if follow_up:
        follow_up += (
            "\n\nThis is a follow-up. Apply what the user asked to change, keep what still "
            "holds from the previous answer, and briefly say what changed."
        )

    return f"""
{"Generate a final travel plan" if is_trip else "Answer the user's travel question"} based on the following information:
User Query: {state['user_query']}
Trip Details:
{format_trip_details(state.get('trip'))}
{data}{follow_up}

{FULL_PLAN_INSTRUCTIONS if is_trip else FOCUSED_INSTRUCTIONS}
- {reply_language_instruction(state.get('trip'))}
"""


async def final_synthesizer(state: TravelState) -> dict:
    """Format all agent outputs into a polished, user-facing answer."""
    logger.info("Final synthesizer: formatting results (intent=%s)", state.get("user_intent"))

    final_prompt = build_final_prompt(state)

    try:
        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content="You are a professional travel assistant. You turn research gathered by other agents into a clear final answer for the user."),
            HumanMessage(content=final_prompt),
        ])
    except Exception as e:  # noqa: BLE001
        logger.error("Final synthesizer failed: %s", e)
        return {
            "messages": [
                AIMessage(
                    content="I apologize, but I encountered an error generating your travel plan. "
                    "Please try again or refine your request."
                )
            ],
            "errors": [f"final_synthesizer: {e!s}"],
        }

    return {
        "messages": [response],
        "llm_calls": 1,
    }
