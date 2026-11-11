"""Rate limiting scope/keying, JSON logging opt-in, and reply-language handling."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import navigo_agent
import navigo_agent.graph.slots as slots_mod
from navigo_agent import middleware
from navigo_agent.graph.agents.final import build_final_prompt
from navigo_agent.graph.slots import (
    TripSlots,
    format_trip_details,
    localise_question,
    parse_trip_extraction,
    reply_language_instruction,
)
from navigo_agent.logging_config import configure_logging_from_env
from navigo_agent.state import build_initial_state

OK_RESULT = {
    "thread_id": "user_abc", "answer": "plan", "needs_clarification": False, "follow_up": False,
    "flight_results": "", "hotel_results": "", "weather_results": "", "itinerary": "", "llm_calls": 1,
}


@pytest.fixture
def client(monkeypatch):
    from app import app

    monkeypatch.setattr(middleware, "RATE_LIMIT", "3/minute")
    with patch("app.run_travel_agent", AsyncMock(return_value=OK_RESULT)):
        yield TestClient(app)


def _post(client, path="/api/travel", **headers):
    return client.post(path, json={"message": "Plan a trip to Tokyo", "thread_id": None}, headers=headers)


class TestRateLimitScope:
    def test_health_page_and_static_are_never_limited(self, client):
        for path in ("/health", "/", "/static/style.css"):
            codes = {client.get(path).status_code for _ in range(12)}
            assert codes == {200}, path

    def test_api_is_limited_with_json_error(self, client):
        assert [_post(client).status_code for _ in range(3)] == [200, 200, 200]
        blocked = _post(client)
        assert blocked.status_code == 429
        body = blocked.json()
        assert body["success"] is False
        assert "Too many requests" in body["error"]
        assert blocked.headers["Retry-After"] == "60"
        # Health checks keep working while the API is throttled
        assert client.get("/health").status_code == 200

    def test_both_travel_endpoints_share_one_bucket(self, client):
        async def fake_stream(**_kwargs):
            yield "event: complete\ndata: {}\n\n"

        with patch("app.stream_travel_plan", fake_stream):
            assert _post(client).status_code == 200
            assert _post(client, "/api/travel/stream").status_code == 200
            assert _post(client).status_code == 200
            assert _post(client, "/api/travel/stream").status_code == 429


class TestClientIp:
    def _request(self, forwarded=None, peer="10.0.0.1"):
        headers = {"x-forwarded-for": forwarded} if forwarded else {}
        return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer))

    def test_header_ignored_by_default(self, monkeypatch):
        monkeypatch.delenv("TRUSTED_PROXY_COUNT", raising=False)
        assert middleware.client_ip(self._request("1.1.1.1, 2.2.2.2")) == "10.0.0.1"

    @pytest.mark.parametrize("count,expected", [("1", "2.2.2.2"), ("2", "1.1.1.1")])
    def test_counts_trusted_proxies_from_the_right(self, monkeypatch, count, expected):
        """Entries left of the trusted ones are client-supplied and never used."""
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", count)
        assert middleware.client_ip(self._request("6.6.6.6, 1.1.1.1, 2.2.2.2")) == expected

    @pytest.mark.parametrize("forwarded", [None, "", " , "])
    def test_falls_back_to_peer_when_header_missing_or_short(self, monkeypatch, forwarded):
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
        assert middleware.client_ip(self._request(forwarded)) == "10.0.0.1"

    def test_bad_env_value_means_no_trust(self, monkeypatch):
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", "many")
        assert middleware.client_ip(self._request("1.1.1.1")) == "10.0.0.1"

    def test_users_behind_a_proxy_get_separate_buckets(self, client, monkeypatch):
        monkeypatch.setenv("TRUSTED_PROXY_COUNT", "1")
        alice = {"X-Forwarded-For": "1.1.1.1"}
        bob = {"X-Forwarded-For": "2.2.2.2"}
        assert [_post(client, **alice).status_code for _ in range(4)] == [200, 200, 200, 429]
        assert _post(client, **bob).status_code == 200


class TestLoggingOptIn:
    @pytest.fixture(autouse=True)
    def _restore_root_logger(self):
        root = logging.getLogger()
        handlers, level = list(root.handlers), root.level
        yield
        root.handlers[:] = handlers
        root.setLevel(level)

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        before = list(logging.getLogger().handlers)
        assert configure_logging_from_env() is False
        assert logging.getLogger().handlers == before

    def test_json_logs_when_enabled(self, monkeypatch, capsys):
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("LOG_LEVEL", "warning")
        assert configure_logging_from_env() is True
        assert logging.getLogger().level == logging.WARNING
        logging.getLogger("navigo.test").warning("hello", extra={"thread_id": "t1"})
        record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert record["message"] == "hello"
        assert record["thread_id"] == "t1"


class TestLanguage:
    def test_language_slot_is_extracted_and_listed(self):
        _, slots = parse_trip_extraction('{"intent": "weather", "destination": "东京", "language": "Chinese"}')
        assert slots.language == "Chinese"
        assert "- User's language: Chinese" in format_trip_details(slots.model_dump())

    @pytest.mark.parametrize("language", [None, "English", "english", "en"])
    def test_english_or_unknown_needs_no_translation(self, language):
        assert reply_language_instruction({"language": language}) == (
            "Write the answer in the same language as the user's query."
        )

    def test_other_language_is_named_in_final_prompt(self):
        state = build_initial_state("东京天气怎么样")
        state.update(user_intent="weather", weather_results="Sunny", trip={"language": "Chinese"})
        assert "Write the entire answer in Chinese" in build_final_prompt(state)

    @pytest.mark.asyncio
    async def test_localise_question_skips_llm_for_english(self):
        with patch.object(slots_mod, "get_llm", side_effect=AssertionError("no LLM call expected")):
            assert await localise_question("Where to?", "English") == "Where to?"
            assert await localise_question("Where to?", None) == "Where to?"

    @pytest.mark.asyncio
    async def test_localise_question_translates(self):
        llm = AsyncMock()
        llm.ainvoke.return_value = AIMessage(content=" 您想去哪里？ ")
        with patch.object(slots_mod, "get_llm", lambda **_: llm):
            assert await localise_question("Where would you like to go?", "Chinese") == "您想去哪里？"
        system, human = llm.ainvoke.await_args.args[0]
        assert "Chinese" in system.content
        assert human.content == "Where would you like to go?"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", [RuntimeError("down"), AIMessage(content="  ")])
    async def test_localise_question_falls_back_to_english(self, outcome):
        llm = AsyncMock()
        if isinstance(outcome, Exception):
            llm.ainvoke.side_effect = outcome
        else:
            llm.ainvoke.return_value = outcome
        with patch.object(slots_mod, "get_llm", lambda **_: llm):
            assert await localise_question("Where to?", "Chinese") == "Where to?"

    @pytest.mark.asyncio
    async def test_clarification_is_asked_in_the_users_language_once(self, graph_env):
        import navigo_agent.graph.routing as routing_mod

        graph_env["extract"].side_effect = [
            ("weather", TripSlots(language="Chinese")),
            ("weather", TripSlots(destination="Tokyo", language="Chinese")),
        ]
        translate = AsyncMock(return_value="您想去哪里？")
        with patch.object(routing_mod, "localise_question", translate):
            first = await navigo_agent.run_travel_agent("天气怎么样？")
            assert first["needs_clarification"] is True
            assert first["question"] == "您想去哪里？"

            second = await navigo_agent.run_travel_agent("东京", thread_id=first["thread_id"])
            assert second["needs_clarification"] is False

        # Translated once — not again when the paused node re-ran on resume
        assert translate.await_count == 1
        assert translate.await_args.args[1] == "Chinese"
        assert "Write the entire answer in Chinese" in graph_env["final_prompts"][-1]
