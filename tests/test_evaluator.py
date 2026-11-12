"""Tests for the evaluation harness (agent and judge are mocked)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from navigo_agent.evaluation import evaluator
from navigo_agent.evaluation.evaluator import (
    DEFAULT_SCOPE,
    EvalResult,
    agent_accuracy,
    expected_scope,
    main,
    run_single_eval,
    select_cases,
    write_report,
)
from navigo_agent.evaluation.test_cases import EVALUATION_CASES

ALL_FOUR = ["flight_agent", "hotel_agent", "weather_agent", "itinerary_agent"]
LONG_ANSWER = "Here is your plan. " * 20


def _agent_result(**overrides) -> dict:
    result = {
        "thread_id": "t-1",
        "answer": LONG_ANSWER,
        "intent": "weather",
        "agents": ["weather_agent"],
        "errors": [],
        "llm_calls": 3,
        "follow_up": False,
    }
    result.update(overrides)
    return result


def _judge(relevance=5, completeness=4, faithfulness=4):
    scores = {
        "relevance": relevance,
        "completeness": completeness,
        "faithfulness": faithfulness,
        "summary": "ok",
    }
    return AsyncMock(return_value=(round((relevance + completeness + faithfulness) / 3), scores))


WEATHER_CASE = {
    "id": "w1",
    "category": "weather",
    "query": "Weather in Tokyo?",
    "expected_intent": "weather",
    "expected_agents": ["weather_agent"],
}


class TestCases:
    def test_ids_are_unique(self):
        ids = [case["id"] for case in EVALUATION_CASES]
        assert len(ids) == len(set(ids))

    def test_expected_agents_are_known(self):
        for case in EVALUATION_CASES:
            for agent in case["expected_agents"] or []:
                assert agent in evaluator.ALL_AGENTS, case["id"]

    def test_select_by_id_and_category(self):
        assert [c["id"] for c in select_cases(case_ids=["weather_only_001"])] == ["weather_only_001"]
        assert {c["category"] for c in select_cases(categories=["weather"])} == {"weather"}
        assert select_cases() == EVALUATION_CASES

    def test_unknown_id_is_rejected(self):
        with pytest.raises(ValueError, match="nope"):
            select_cases(case_ids=["nope"])


class TestScoring:
    def test_agent_accuracy(self):
        assert agent_accuracy(["weather_agent"], ["weather_agent"]) == 1.0
        assert agent_accuracy(["weather_agent"], ALL_FOUR) == 0.25
        assert agent_accuracy(ALL_FOUR, ["flight_agent", "hotel_agent", "weather_agent"]) == 0.75
        assert agent_accuracy(None, ALL_FOUR) == 1.0

    def test_scope_follows_intent(self):
        assert "not expected" in expected_scope({"expected_intent": "weather"})
        assert "itinerary" in expected_scope({"expected_intent": "full_trip"})
        assert expected_scope({"expected_intent": "general"}) == DEFAULT_SCOPE

    @pytest.mark.asyncio
    async def test_judge_prompt_carries_the_scope(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='Sure: {"relevance": 5, "completeness": 4, "faithfulness": 3, "summary": "s"}'
        ))
        with patch.object(evaluator, "get_llm", return_value=llm):
            overall, scores = await evaluator._llm_quality_score("q", LONG_ANSWER, "ONLY WEATHER")
        assert overall == 4
        assert scores["faithfulness"] == 3
        assert "ONLY WEATHER" in llm.ainvoke.call_args.args[0]

    @pytest.mark.asyncio
    async def test_judge_failure_is_flagged(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("down"))
        with patch.object(evaluator, "get_llm", return_value=llm):
            _, scores = await evaluator._llm_quality_score("q", LONG_ANSWER)
        assert scores["judge_failed"] is True


class TestRunSingleEval:
    @pytest.mark.asyncio
    async def test_passing_case(self):
        agent = AsyncMock(return_value=_agent_result())
        with patch("navigo_agent.run_travel_agent", agent), \
             patch.object(evaluator, "_llm_quality_score", _judge()) as judge:
            result = await run_single_eval(WEATHER_CASE)

        assert result.passed
        assert result.tool_accuracy == 1.0
        assert result.intent_correct is True
        assert result.agents == ["weather_agent"]
        assert agent.call_args.kwargs["allow_clarification"] is False
        assert "not expected" in judge.call_args.args[2]

    @pytest.mark.asyncio
    async def test_wrong_agents_fail_and_wrong_intent_is_noted(self):
        agent = AsyncMock(return_value=_agent_result(intent="full_trip", agents=ALL_FOUR))
        with patch("navigo_agent.run_travel_agent", agent), \
             patch.object(evaluator, "_llm_quality_score", _judge()):
            result = await run_single_eval(WEATHER_CASE)

        assert not result.passed
        assert result.tool_accuracy == 0.25
        assert result.intent_correct is False
        assert "intent='full_trip'" in result.notes

    @pytest.mark.asyncio
    async def test_agent_exception_is_a_failed_case(self):
        agent = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("navigo_agent.run_travel_agent", agent):
            result = await run_single_eval(WEATHER_CASE)
        assert not result.passed
        assert "boom" in result.notes

    @pytest.mark.asyncio
    async def test_follow_up_runs_a_second_turn_on_the_same_thread(self):
        case = {**WEATHER_CASE, "follow_up": "And next week?"}
        agent = AsyncMock(side_effect=[
            _agent_result(thread_id="first"),
            _agent_result(thread_id="second", answer="Second answer. " * 20, follow_up=True, agents=[]),
        ])
        with patch("navigo_agent.run_travel_agent", agent), \
             patch.object(evaluator, "_llm_quality_score", _judge()) as judge:
            result = await run_single_eval(case)

        assert result.passed
        assert agent.call_args_list[1].args == ("And next week?",)
        assert agent.call_args_list[1].kwargs["thread_id"] == "first"
        # Agents are judged on the first turn, quality on the second answer.
        assert result.agents == ["weather_agent"]
        assert judge.call_args.args[1].startswith("Second answer.")
        assert "llm_calls=6" in result.notes

    @pytest.mark.asyncio
    async def test_follow_up_not_recognised_fails(self):
        case = {**WEATHER_CASE, "follow_up": "And next week?"}
        agent = AsyncMock(side_effect=[_agent_result(), _agent_result(follow_up=False)])
        with patch("navigo_agent.run_travel_agent", agent), \
             patch.object(evaluator, "_llm_quality_score", _judge()):
            result = await run_single_eval(case)
        assert not result.passed
        assert "not treated as a follow-up" in result.notes


class TestReportAndCli:
    def test_write_report(self, tmp_path):
        path = tmp_path / "eval.json"
        write_report([
            EvalResult(test_id="a", category="weather", passed=True, agents=["weather_agent"]),
            EvalResult(test_id="b", category="hotel", passed=False),
        ], path)
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["results"][0]["agents"] == ["weather_agent"]

    def test_main_runs_selected_cases_and_writes_report(self, tmp_path, capsys):
        path = tmp_path / "eval.json"
        agent = AsyncMock(return_value=_agent_result())
        with patch("navigo_agent.run_travel_agent", agent), \
             patch.object(evaluator, "_llm_quality_score", _judge()):
            code = main(["--case", "weather_only_001", "--output", str(path)])

        assert code == 0
        assert agent.await_count == 1
        assert json.loads(path.read_text(encoding="utf-8"))["total"] == 1
        assert "Intent accuracy:    1/1" in capsys.readouterr().out

    def test_main_rejects_unknown_case(self):
        with pytest.raises(SystemExit):
            main(["--case", "nope"])
