"""Graph builder: constructs the supervisor-driven LangGraph state graph.

Architecture:
  START → intent_classifier → supervisor → [conditional routing]
      ├── flight_agent    ──┐
      ├── hotel_agent     ──┤ (parallel via Send API)
      ├── weather_agent   ──┘
      ├── itinerary_agent   (sequential: needs flight+hotel+weather data)
      └── final_synthesizer → END
           ↑
      supervisor (loop back, max 3 rounds)
"""

import asyncio
import logging
import selectors

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row

from navigo_agent.config import get_checkpointer_dsn
from navigo_agent.graph.agents.final import final_synthesizer
from navigo_agent.graph.agents.flight import flight_agent
from navigo_agent.graph.agents.hotel import hotel_agent
from navigo_agent.graph.agents.itinerary import itinerary_agent
from navigo_agent.graph.agents.weather import weather_agent
from navigo_agent.graph.routing import (
    intent_classifier,
    route_after_agent,
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
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_synthesizer", final_synthesizer)

    # ── Edges ────────────────────────────────────────────────────────
    # Entry: always classify intent first
    graph.add_edge(START, "intent_classifier")

    # Intent → Supervisor
    graph.add_edge("intent_classifier", "supervisor")

    # Supervisor → Conditional: which agents to dispatch
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

    # Each agent → Conditional: more agents or back to supervisor
    for agent_node in ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"]:
        graph.add_conditional_edges(
            agent_node,
            route_after_agent,
            {
                "flight_agent": "flight_agent",
                "hotel_agent": "hotel_agent",
                "weather_agent": "weather_agent",
                "itinerary_agent": "itinerary_agent",
                "supervisor": "supervisor",
                "final_synthesizer": "final_synthesizer",
            },
        )

    # Final synthesizer → END
    graph.add_edge("final_synthesizer", END)

    return graph


# ── Checkpointer Initialization ──────────────────────────────────────
#
# IMPORTANT: asyncio.run() MUST happen at module import time, NOT lazily
# on the first request.  When lazy-init fires inside a FastAPI handler,
# Uvicorn's event loop is already running and Python ≥3.10 raises:
#   RuntimeError: asyncio.run() cannot be called from a running event loop
#
# Module-level init runs during `app.py` import (before Uvicorn starts)
# so there is no event loop conflict.

_checkpointer: AsyncPostgresSaver | None = None


async def _init_checkpointer():
    """Create async Postgres connection and checkpointer."""
    dsn = get_checkpointer_dsn()
    async_conn = await psycopg.AsyncConnection.connect(
        dsn,
        autocommit=True,
        row_factory=dict_row,
    )
    cp = AsyncPostgresSaver(async_conn)
    await cp.setup()
    return cp


def _eager_init_checkpointer() -> None:
    """Initialise the module-level checkpointer synchronously.

    Called at import time — before any event loop exists.
    """
    global _checkpointer
    try:
        _checkpointer = asyncio.run(
            _init_checkpointer(),
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
        logger.info("PostgreSQL checkpointer initialised (eager, import-time).")
    except ValueError as error:
        if "DATABASE_URL is missing" in str(error):
            logger.warning(
                "Skipping PostgreSQL checkpointer initialisation at import time "
                "because DATABASE_URL is not set."
            )
            return
        raise
    except Exception:
        logger.exception(
            "Failed to initialise PostgreSQL checkpointer at import time. "
            "Check DATABASE_URL in .env and network connectivity."
        )
        raise  # fail fast — app cannot serve requests without persistence


_eager_init_checkpointer()


def get_checkpointer() -> AsyncPostgresSaver:
    """Return the module-level checkpointer (already initialised)."""
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer was not initialised. Ensure DATABASE_URL is set and startup "
            "checkpointer initialization succeeds."
        )
    return _checkpointer


# ── Compiled Graph (module-level singleton) ──────────────────────────

_compiled_graph = None


def get_compiled_graph():
    """Return the compiled graph with PostgreSQL checkpointer."""
    global _compiled_graph
    if _compiled_graph is None:
        checkpointer = get_checkpointer()
        raw_graph = build_travel_graph()
        _compiled_graph = raw_graph.compile(checkpointer=checkpointer)
        logger.info("Graph compiled with PostgreSQL checkpointer.")
    return _compiled_graph
