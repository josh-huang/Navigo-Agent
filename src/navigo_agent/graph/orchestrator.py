"""Orchestrator: the main agent of the travel graph.

One agent owns the request from the moment intake is done until the answer is
written. It works in a tool-calling loop:

    orchestrator ──tool calls──► sub-agents (parallel, via Send) ──reports──► orchestrator ─► … ─► final answer

  - Its tools ARE the sub-agents (see agents/delegation.py). A tool call is a
    delegation: the sub-agent named by the call runs as its own graph node and
    answers with a ToolMessage holding its report.
  - It reads those reports, decides whether more work is needed, and when it
    stops calling tools its reply is the final answer — there is no separate
    synthesizer.

The model decides; a small deterministic plan guard keeps a bad decision from
turning into a bad answer (see `_required_delegations`).
"""

import logging
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Send

from navigo_agent.config import MAX_ORCHESTRATOR_ROUNDS, get_llm
from navigo_agent.graph.agents.delegation import (
    PLANNING_AGENTS,
    RESEARCH_AGENTS,
    SUBAGENT_BY_NAME,
    SUBAGENT_NAMES,
    SUBAGENTS,
    delegation_tools,
)
from navigo_agent.graph.slots import (
    conversation_block,
    format_trip_details,
    reply_language_instruction,
)
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)

# Intents that end in a day-by-day plan (and get the full sectioned answer).
TRIP_INTENTS = frozenset({"full_trip", "itinerary"})


# ── Prompts ───────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = f"""You are Navigo, the main agent of a travel planning assistant.

You own the user's request end to end: you decide what has to be found out, delegate that work to your specialist sub-agents, read their reports, and write the final answer yourself.

## Your Sub-Agents

Each sub-agent is a tool. Calling the tool delegates the work; the tool result is the sub-agent's report.

{chr(10).join(f"- {spec.name}: {spec.description}" for spec in SUBAGENTS)}

## How To Delegate

- Delegate only what the request needs:
  - a weather question → weather_agent only
  - a flight question → flight_agent only
  - a hotel question → hotel_agent only
  - a full trip or itinerary → flight_agent + hotel_agent + weather_agent together, then itinerary_agent, then the answer
- Call every research sub-agent the request needs in ONE turn — they run in parallel. Do not call them one per turn.
- itinerary_agent builds on the other reports: call it in a later turn, once they are in. Never call it for a weather-only, flight-only or hotel-only question.
- Give each sub-agent a short, specific brief. They also receive the trip details, so the brief is for focus: what matters for this user, and for a follow-up, what should change.
- Each sub-agent can be used at most once per request. If a report says information was unavailable, carry on without it — do not call the sub-agent again.
- A general travel question you can answer well yourself needs no sub-agent.

## How To Finish

When you have what you need, stop calling tools and reply with the final answer in markdown, following the "Final Answer" instructions in the request. Do not narrate what you are about to do — either call tools or write the final answer."""

FULL_PLAN_INSTRUCTIONS = """Write a complete travel plan, formatted beautifully with these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical. Keep the plan useful for real travel planning.
- Mention that live ticket prices may not be available if the flight report has no pricing.
- If a section has no report behind it, say the information was unavailable rather than fabricating it."""

FOCUSED_INSTRUCTIONS = """Answer the user's question directly.

Important:
- Cover only what the user asked about. Do NOT add sections for topics they did not
  ask about (no flight, hotel, itinerary or budget sections unless requested).
- Lead with the answer, then the supporting details. Use markdown headings and
  bullet points where they help; keep it concise."""

GROUNDING_INSTRUCTIONS = (
    "Base facts on your sub-agents' reports. For anything they did not cover you may use general "
    "travel knowledge, but never invent specific prices, flight numbers, schedules or opening "
    "hours — say the information was unavailable instead."
)

FINAL_ANSWER_PROMPT = (
    "Write the final answer for the user now, in markdown, following the Final Answer "
    "instructions and using the reports above. Do not call any tools."
)
APOLOGY_ANSWER = (
    "I apologize, but I encountered an error generating your travel plan. "
    "Please try again or refine your request."
)


def build_request_brief(state: TravelState) -> str:
    """The orchestrator's view of the request: what was asked, what is known, how to answer.

    Rebuilt on every turn from the state, so the progress section is always
    current. The sub-agents' reports are not repeated here — they follow as
    ToolMessages in the transcript.
    """
    is_trip = state.get("user_intent") in TRIP_INTENTS
    completed = list(state.get("completed_agents") or [])
    unused = [name for name in SUBAGENT_NAMES if name not in completed]
    errors = list(state.get("errors") or [])
    rounds = state.get("orchestrator_rounds", 0)

    # Follow-up: show the previous answer at length so it is revised, not restarted.
    follow_up = conversation_block(state.get("history"), answer_chars=1500)
    if follow_up:
        follow_up += (
            "\n\nThis is a follow-up. Delegate again whatever has to be redone, apply what the "
            "user asked to change, keep what still holds from the previous answer, and briefly "
            "say what changed."
        )

    return f"""## User Request

{state.get('user_query', '')}

## Trip Details (extracted from the request)

- Intent: {state.get('user_intent') or 'general'}
{format_trip_details(state.get('trip'))}{follow_up}

## Progress

- Reports received from: {', '.join(completed) or 'none yet'}
- Sub-agents not used yet: {', '.join(unused) or 'none'}
- Delegation rounds used: {rounds} of {MAX_ORCHESTRATOR_ROUNDS}
- Problems so far: {'; '.join(errors) or 'none'}

## Final Answer

{FULL_PLAN_INSTRUCTIONS if is_trip else FOCUSED_INSTRUCTIONS}
- {GROUNDING_INSTRUCTIONS}
- {reply_language_instruction(state.get('trip'))}"""


def _transcript(state: TravelState) -> list:
    """The delegation history: the orchestrator's tool calls and the sub-agents' reports."""
    return [
        message
        for message in state.get("messages") or []
        if isinstance(message, ToolMessage) or (isinstance(message, AIMessage) and message.tool_calls)
    ]


def build_orchestrator_messages(state: TravelState) -> list:
    """What the orchestrator LLM sees on a turn: instructions, the request, the work so far."""
    return [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=build_request_brief(state)),
        *_transcript(state),
    ]


def _text(response) -> str:
    """Plain text of an AIMessage (content may be a list of blocks)."""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content or "").strip()


# ── Plan guard ────────────────────────────────────────────────────────
#
# What a sound orchestrator would delegate for each intent. Applied when the
# model cannot be trusted for this turn: the call failed, its output was
# unusable, or it tried to answer before anything was gathered. Without it,
# one model hiccup produced an answer built on no data.

DEFAULT_PLAN: dict[str, tuple[str, ...]] = {
    "weather": ("weather_agent",),
    "flight": ("flight_agent",),
    "flights": ("flight_agent",),
    "hotel": ("hotel_agent",),
    "full_trip": RESEARCH_AGENTS,
    "itinerary": RESEARCH_AGENTS,
    "general": (),
}


def default_plan(state: TravelState) -> list[str]:
    """Research sub-agents the intent calls for that have not reported yet."""
    completed = set(state.get("completed_agents") or [])
    planned = DEFAULT_PLAN.get(state.get("user_intent", ""), DEFAULT_PLAN["full_trip"])
    return [agent for agent in planned if agent not in completed]


def _required_delegations(state: TravelState, *, decided: bool) -> list[str]:
    """Delegations the request still needs although the model issued none.

    `decided` is False when the model gave nothing usable this turn (the call
    failed, or it returned neither a valid delegation nor an answer).
    """
    completed = set(state.get("completed_agents") or [])
    nothing_gathered = not completed and not any(
        state.get(spec.result_key) for spec in SUBAGENTS
    )

    planned = default_plan(state)
    if planned and (not decided or nothing_gathered):
        return planned

    # A trip plan needs its itinerary: once research is in, it is always built.
    has_research = any(state.get(SUBAGENT_BY_NAME[name].result_key) for name in RESEARCH_AGENTS)
    if state.get("user_intent") in TRIP_INTENTS and has_research:
        return [name for name in PLANNING_AGENTS if name not in completed]

    return []


def _select_delegations(tool_calls: list[dict] | None, completed: set[str]) -> list[dict]:
    """Keep the tool calls that are valid delegations for this round.

    - unknown tools are dropped (the model may only call its sub-agents)
    - each sub-agent runs at most once per request, and once per round
    - planning sub-agents consume the research reports, so they never share a
      round with research sub-agents: they are held back until the next one
    """
    selected: dict[str, dict] = {}
    for call in tool_calls or []:
        name = call.get("name")
        if name not in SUBAGENT_BY_NAME:
            logger.warning("Orchestrator called unknown tool '%s' — ignored.", name)
        elif name in completed:
            logger.info("Orchestrator re-delegated to completed sub-agent '%s' — ignored.", name)
        elif name not in selected:
            args = call.get("args")
            selected[name] = _tool_call(name, args if isinstance(args, dict) else {}, call.get("id"))

    if any(name in RESEARCH_AGENTS for name in selected):
        for name in PLANNING_AGENTS:
            if selected.pop(name, None):
                logger.info("'%s' held back until the research reports are in.", name)
    return list(selected.values())


def _tool_call(name: str, args: dict | None = None, call_id: str | None = None) -> dict:
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "tool_call",
    }


# ── Orchestrator Node ─────────────────────────────────────────────────

async def orchestrator_node(state: TravelState) -> dict:
    """One turn of the main agent: delegate to sub-agents, or write the final answer.

    Runs after intake and again after every round of sub-agent reports. The
    returned message is either an AIMessage with tool calls (→ the graph fans
    out to those sub-agents) or the final answer (→ the run ends).
    """
    completed = set(state.get("completed_agents") or [])
    rounds = state.get("orchestrator_rounds", 0)
    can_delegate = rounds < MAX_ORCHESTRATOR_ROUNDS and any(n not in completed for n in SUBAGENT_NAMES)

    llm = get_llm(temperature=0.2)
    # Every sub-agent stays declared on every turn: the transcript refers to them.
    tools = delegation_tools()
    conversation = build_orchestrator_messages(state)

    llm_calls = 0
    errors: list[str] = []
    delegations: list[dict] = []
    answer = ""
    decided = True

    # ── Decide: delegate more work, or answer ────────────────────────
    if can_delegate:
        try:
            response = await llm.bind_tools(tools).ainvoke(conversation)
            llm_calls += 1
        except Exception as e:  # noqa: BLE001
            # e.g. the provider rejecting a malformed tool call
            logger.error("Orchestrator LLM call failed: %s", e)
            errors.append(f"orchestrator_llm_error: {e!s}")
            decided = False
        else:
            requested = getattr(response, "tool_calls", None) or []
            delegations = _select_delegations(requested, completed)
            answer = "" if requested else _text(response)
            decided = bool(delegations or answer)
    else:
        logger.info("Orchestrator: nothing left to delegate (round %d) — answering.", rounds)

    # ── Plan guard: the model issued no delegation, but the request needs one ──
    if not delegations:
        required = _required_delegations(state, decided=decided)
        if required:
            logger.warning("Orchestrator plan guard — delegating to %s.", required)
            delegations = [_tool_call(name) for name in required]
            answer = ""  # written before the data existed

    if delegations:
        logger.info("Orchestrator round %d: delegating to %s", rounds + 1, [c["name"] for c in delegations])
        return {
            "messages": [AIMessage(content="", tool_calls=delegations)],
            "orchestrator_rounds": rounds + 1,
            "llm_calls": llm_calls,
            "errors": errors,
        }

    # ── Final answer ─────────────────────────────────────────────────
    if not answer:
        conversation.append(HumanMessage(content=FINAL_ANSWER_PROMPT))
        try:
            response = await llm.bind_tools(tools, tool_choice="none").ainvoke(conversation)
            llm_calls += 1
            answer = _text(response)
        except Exception as e:  # noqa: BLE001
            logger.error("Orchestrator final answer failed: %s", e)
            errors.append(f"orchestrator_final_answer: {e!s}")

    logger.info("Orchestrator: final answer after %d delegation round(s).", rounds)
    return {
        "messages": [AIMessage(content=answer or APOLOGY_ANSWER)],
        "llm_calls": llm_calls,
        "errors": errors,
    }


# ── Routing (conditional edge after the orchestrator) ─────────────────

def pending_delegations(state: TravelState) -> list[dict]:
    """Tool calls of the orchestrator's last message — empty once it has answered."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    return list(last.tool_calls) if isinstance(last, AIMessage) else []


def route_after_orchestrator(state: TravelState) -> list[Send] | str:
    """Fan out to the delegated sub-agents, or end the run.

    One `Send` per tool call: LangGraph runs them concurrently in one
    superstep and the orchestrator is triggered again only once all of them
    have reported. Each sub-agent receives the state plus its own delegation.
    """
    delegations = pending_delegations(state)
    if not delegations:
        return END
    return [
        Send(call["name"], {**state, "delegation": {"id": call["id"], "args": call["args"]}})
        for call in delegations
    ]
