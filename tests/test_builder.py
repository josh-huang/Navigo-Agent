"""Checkpointer lifecycle tests — no database needed."""

import pytest
from langgraph.checkpoint.memory import MemorySaver

from navigo_agent.graph import builder


@pytest.fixture(autouse=True)
def _reset_builder_state():
    saved = (builder._checkpointer, builder._pool, builder._compiled_graph)
    builder._checkpointer = builder._pool = builder._compiled_graph = None
    yield
    builder._checkpointer, builder._pool, builder._compiled_graph = saved


def test_importing_app_does_not_touch_the_database(monkeypatch):
    """Nothing connects at import time — only the lifespan does."""
    import app  # noqa: F401

    assert builder._pool is None


@pytest.mark.asyncio
async def test_init_without_database_url_uses_memory(monkeypatch):
    monkeypatch.setattr(builder, "DATABASE_URL", None)
    saver = await builder.init_checkpointer()
    assert isinstance(saver, MemorySaver)
    assert builder._pool is None
    # Idempotent
    assert await builder.init_checkpointer() is saver
    # The graph is compiled against that checkpointer
    assert builder.get_compiled_graph().checkpointer is saver


@pytest.mark.asyncio
async def test_close_resets_state(monkeypatch):
    monkeypatch.setattr(builder, "DATABASE_URL", None)
    await builder.init_checkpointer()
    builder.get_compiled_graph()
    await builder.close_checkpointer()
    assert builder._checkpointer is None
    assert builder._compiled_graph is None
    await builder.close_checkpointer()  # safe to call twice


def test_graph_compiles_without_init():
    """Scripts and the evaluator call get_compiled_graph() without the FastAPI lifespan."""
    graph = builder.get_compiled_graph()
    assert isinstance(graph.checkpointer, MemorySaver)
    assert builder.get_compiled_graph() is graph


@pytest.mark.asyncio
async def test_unreachable_database_fails_fast(monkeypatch):
    """A configured but unreachable database is a deploy error, not a silent fallback."""
    monkeypatch.setattr(builder, "DATABASE_URL", "postgresql://u:p@127.0.0.1:1/db")
    monkeypatch.setattr(builder, "get_checkpointer_dsn", lambda: "postgresql://u:p@127.0.0.1:1/db")

    class FailingPool:
        def __init__(self, **_kwargs):
            self.closed = False

        async def open(self, **_kwargs):
            raise ConnectionError("connection refused")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(builder, "AsyncConnectionPool", FailingPool)
    with pytest.raises(ConnectionError):
        await builder.init_checkpointer()
    assert builder._checkpointer is None
    assert builder._pool is None


def test_lifespan_initialises_and_closes(monkeypatch):
    from fastapi.testclient import TestClient

    from app import app

    monkeypatch.setattr(builder, "DATABASE_URL", None)
    with TestClient(app) as client:
        assert isinstance(builder._checkpointer, MemorySaver)
        assert client.get("/health").status_code == 200
    assert builder._checkpointer is None
