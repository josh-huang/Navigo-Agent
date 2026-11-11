"""Graph builder: constructs the supervisor-driven LangGraph state graph.

Architecture:
  START → intent_classifier → clarify (may pause to ask the user) → supervisor → [conditional routing]
      ├── flight_agent    ──┐
      ├── hotel_agent     ──┤ parallel fan-out (same superstep) → back to supervisor
      ├── weather_agent   ──┘ (max 3 supervisor rounds)
      ├── itinerary_agent ──→ final_synthesizer   (needs flight+hotel+weather data)
      └── final_synthesizer → END
"""

import asyncio
import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from navigo_agent.config import DATABASE_URL, get_checkpointer_dsn
from navigo_agent.graph.agents.final import final_synthesizer
from navigo_agent.graph.agents.flight import flight_agent
from navigo_agent.graph.agents.hotel import hotel_agent
from navigo_agent.graph.agents.itinerary import itinerary_agent
from navigo_agent.graph.agents.weather import weather_agent
from navigo_agent.graph.routing import (
    PARALLEL_AGENTS,
    clarify_node,
    intent_classifier,
    prepare_clarification,
    route_after_supervisor,
)
from navigo_agent.graph.supervisor import supervisor_node
from navigo_agent.state import TravelState

logger = logging.getLogger(__name__)


# ── Graph Construction ───────────────────────────────────────────────

def build_travel_graph() -> StateGraph:
    """Build and return the compiled supervisor-orchestrated travel graph.

    Does NOT compile with checkpointer — that happens separately so
    tests can use MemorySaver.
    """
    graph = StateGraph(TravelState)

    # ── Add nodes ────────────────────────────────────────────────────
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("prepare_clarification", prepare_clarification)
    graph.add_node("clarify", clarify_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_synthesizer", final_synthesizer)

    # ── Edges ────────────────────────────────────────────────────────
    # Entry: always classify intent first
    graph.add_edge(START, "intent_classifier")

    # Intent + slots → clarify (interrupts if essentials are missing) → Supervisor
    graph.add_edge("intent_classifier", "prepare_clarification")
    graph.add_edge("prepare_clarification", "clarify")
    graph.add_edge("clarify", "supervisor")

    # Supervisor → Conditional: one node, or a list of nodes to run in parallel
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "weather_agent": "weather_agent",
            "itinerary_agent": "itinerary_agent",
            "final_synthesizer": "final_synthesizer",
        },
    )

    # Data agents fan in to the supervisor. When several run in parallel the
    # supervisor is triggered ONCE, after the whole batch has finished.
    for agent_node in PARALLEL_AGENTS:
        graph.add_edge(agent_node, "supervisor")

    # The itinerary is the last data-producing step — nothing left to decide.
    graph.add_edge("itinerary_agent", "final_synthesizer")

    # Final synthesizer → END
    graph.add_edge("final_synthesizer", END)

    return graph


# ── Checkpointer lifecycle ───────────────────────────────────────────
#
# The checkpointer is created inside the running event loop — FastAPI's
# lifespan calls init_checkpointer() at startup and close_checkpointer() at
# shutdown. Nothing touches the database at import time, so importing the
# package (tests, scripts, tooling) never needs a reachable Postgres.
#
#   DATABASE_URL set   → AsyncPostgresSaver on a connection pool
#   DATABASE_URL unset → in-memory MemorySaver (state is lost on restart)

POOL_MAX_SIZE = 10

_checkpointer: BaseCheckpointSaver | None = None
_pool: AsyncConnectionPool | None = None
_compiled_graph = None


def _loop_supports_psycopg() -> bool:
    """psycopg's async mode cannot run on Windows' ProactorEventLoop."""
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    return proactor is None or not isinstance(asyncio.get_running_loop(), proactor)


async def init_checkpointer() -> BaseCheckpointSaver:
    """Create the checkpointer. Call once, from inside the server's event loop."""
    global _checkpointer, _pool, _compiled_graph

    if _checkpointer is not None:
        return _checkpointer

    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL is not set — using an in-memory checkpointer. "
            "Paused conversations are lost when the process restarts."
        )
        _checkpointer = MemorySaver()
    elif not _loop_supports_psycopg():
        logger.error(
            "PostgreSQL checkpointing needs a selector event loop, but this process runs "
            "on Windows' ProactorEventLoop. Start the app with `python app.py` (or "
            "`uvicorn app:app --reload`). Falling back to an in-memory checkpointer."
        )
        _checkpointer = MemorySaver()
    else:
        pool = AsyncConnectionPool(
            conninfo=get_checkpointer_dsn(),
            max_size=POOL_MAX_SIZE,
            open=False,
            # Settings AsyncPostgresSaver requires of its connections
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        try:
            await pool.open(wait=True, timeout=30)
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
        except Exception:
            await pool.close()
            logger.exception(
                "Failed to initialise the PostgreSQL checkpointer. "
                "Check DATABASE_URL and network connectivity."
            )
            raise  # fail fast — a configured database that is unreachable is a deploy error
        _pool, _checkpointer = pool, saver
        logger.info("PostgreSQL checkpointer initialised (pool max_size=%d).", POOL_MAX_SIZE)

    _compiled_graph = None  # recompile against the new checkpointer on next use
    return _checkpointer


async def close_checkpointer() -> None:
    """Release the connection pool. Safe to call when nothing was initialised."""
    global _checkpointer, _pool, _compiled_graph
    pool, _pool, _checkpointer, _compiled_graph = _pool, None, None, None
    if pool is not None:
        await pool.close()
        logger.info("PostgreSQL connection pool closed.")


# ── Compiled Graph (module-level singleton) ──────────────────────────

def get_compiled_graph():
    """Return the compiled graph, bound to the active checkpointer.

    If init_checkpointer() was never called (scripts, the evaluator, tests),
    an in-memory checkpointer is used so the graph still works.
    """
    global _compiled_graph, _checkpointer
    if _compiled_graph is None:
        if _checkpointer is None:
            logger.warning(
                "Checkpointer not initialised — compiling the graph with an in-memory one."
            )
            _checkpointer = MemorySaver()
        _compiled_graph = build_travel_graph().compile(checkpointer=_checkpointer)
        logger.info("Graph compiled with %s.", type(_checkpointer).__name__)
    return _compiled_graph
