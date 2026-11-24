"""The contract between the orchestrator and its sub-agents.

The orchestrator delegates by calling a tool named after the sub-agent. The
graph turns each tool call into a `Send(<sub-agent node>, payload)`, where the
payload is the current state plus

    "delegation": {"id": "<tool call id>", "args": {...the tool arguments...}}

A sub-agent does its work and answers with `report()`: a state update whose
ToolMessage carries the report back to the orchestrator under that call id.

`SUBAGENTS` is the single registry: the orchestrator builds its tools from it,
the graph builder its nodes, the streaming layer its per-agent results.
Adding a sub-agent = one spec here + one node in `agents/__init__.py`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from langchain_core.messages import ToolMessage

from navigo_agent.state import TravelState

RESEARCH = "research"  # gathers independent data — research sub-agents run in parallel
PLANNING = "planning"  # builds on the research reports — runs in a later round

_TASK_PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "A short, specific brief for the sub-agent: what to find out or produce and "
                "what to focus on (route, dates, budget, preferences, what changed in a follow-up)."
            ),
        },
    },
    "required": ["task"],
}


@dataclass(frozen=True)
class SubAgentSpec:
    """How the orchestrator sees one sub-agent."""

    name: str  # graph node name == name of the tool the orchestrator calls
    result_key: str  # TravelState key that holds its report
    stage: str  # RESEARCH or PLANNING
    description: str  # shown to the orchestrator as the tool description
    parameters: dict = field(default_factory=lambda: _TASK_PARAMETERS)


SUBAGENTS: tuple[SubAgentSpec, ...] = (
    SubAgentSpec(
        name="flight_agent",
        result_key="flight_results",
        stage=RESEARCH,
        description=(
            "Flight researcher. Searches the web for the route: airports, airlines, flight "
            "duration, typical fares and booking tips. Returns a markdown report."
        ),
    ),
    SubAgentSpec(
        name="hotel_agent",
        result_key="hotel_results",
        stage=RESEARCH,
        description=(
            "Accommodation researcher. Searches the web for the best areas to stay and specific "
            "hotels across budgets, with booking tips. Returns a markdown report."
        ),
    ),
    SubAgentSpec(
        name="weather_agent",
        result_key="weather_results",
        stage=RESEARCH,
        description=(
            "Weather lookup. Fetches the current weather and the 5-day forecast for the "
            "destination city from OpenWeather."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "ONE city to look up, e.g. 'Tokyo' — the trip's main destination city.",
                },
            },
            "required": ["city"],
        },
    ),
    SubAgentSpec(
        name="itinerary_agent",
        result_key="itinerary",
        stage=PLANNING,
        description=(
            "Itinerary planner. Builds a day-by-day plan (attractions, transport, dining) on top "
            "of the flight, hotel and weather reports, searching the web to fill gaps. Call it "
            "only AFTER the research sub-agents have reported — never in the same turn as them."
        ),
    ),
)

SUBAGENT_BY_NAME: dict[str, SubAgentSpec] = {spec.name: spec for spec in SUBAGENTS}
SUBAGENT_NAMES: tuple[str, ...] = tuple(SUBAGENT_BY_NAME)
RESEARCH_AGENTS: tuple[str, ...] = tuple(s.name for s in SUBAGENTS if s.stage == RESEARCH)
PLANNING_AGENTS: tuple[str, ...] = tuple(s.name for s in SUBAGENTS if s.stage == PLANNING)
RESULT_KEYS: tuple[str, ...] = tuple(s.result_key for s in SUBAGENTS)


def delegation_tools(names: Iterable[str] | None = None) -> list[dict]:
    """OpenAI-style tool schemas (accepted by ChatGroq.bind_tools) for the given sub-agents."""
    wanted = SUBAGENT_NAMES if names is None else tuple(names)
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in SUBAGENTS
        if spec.name in wanted
    ]


# ── Sub-agent side ────────────────────────────────────────────────────

def delegation_args(state: TravelState) -> dict:
    """Arguments the orchestrator passed with this delegation ({} when run directly)."""
    args = (state.get("delegation") or {}).get("args")
    return args if isinstance(args, dict) else {}


def task_block(state: TravelState) -> str:
    """A ready-to-append prompt section with the orchestrator's brief ("" if none)."""
    task = str(delegation_args(state).get("task") or "").strip()
    if not task:
        return ""
    return (
        "\n\n## Task From The Orchestrator\n\n"
        "The main agent delegated this to you with the following brief — make sure your "
        f"answer covers it:\n\n{task}"
    )


def report(
    state: TravelState,
    agent: str,
    content: str,
    *,
    llm_calls: int = 0,
    errors: Iterable[str] = (),
) -> dict:
    """The state update a sub-agent returns: deltas only.

    Sub-agents of one round run in the same superstep, so every value here is
    merged by a reducer (or written by this sub-agent alone). The ToolMessage
    is what the orchestrator reads on its next turn.
    """
    call_id = (state.get("delegation") or {}).get("id")
    return {
        SUBAGENT_BY_NAME[agent].result_key: content,
        "messages": [ToolMessage(content=content, tool_call_id=call_id, name=agent)] if call_id else [],
        "llm_calls": llm_calls,
        "errors": list(errors),
        "completed_agents": [agent],
    }
