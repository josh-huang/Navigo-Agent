"""Config tests."""

import inspect
from pathlib import Path

import pytest
import uvicorn

from navigo_agent import config


class TestCheckpointerDsn:
    def test_dsn_is_not_forced_to_ssl(self, monkeypatch):
        """Local Postgres (docker compose) has no SSL — the DSN must pass through."""
        dsn = "postgresql://navigo:navigo@postgres:5432/navigo"
        monkeypatch.setattr(config, "DATABASE_URL", dsn)
        assert config.get_checkpointer_dsn() == dsn

    def test_explicit_sslmode_is_preserved(self, monkeypatch):
        dsn = "postgresql://u:p@host/db?sslmode=require"
        monkeypatch.setattr(config, "DATABASE_URL", dsn)
        assert config.get_checkpointer_dsn() == dsn

    def test_missing_database_url_raises(self, monkeypatch):
        monkeypatch.setattr(config, "DATABASE_URL", None)
        with pytest.raises(ValueError, match="DATABASE_URL is missing"):
            config.get_checkpointer_dsn()


def test_app_entry_point_uses_keyword_args():
    """uvicorn.run() takes host/port/reload as keyword-only arguments."""
    params = inspect.signature(uvicorn.run).parameters
    assert params["host"].kind is inspect.Parameter.KEYWORD_ONLY
    source = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert 'uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)' in source
