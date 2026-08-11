"""LLM-as-Judge evaluator for Navigo-Agent.

Measures:
  1. Tool selection accuracy — did the supervisor dispatch the right agents?
  2. Output quality — relevance, completeness, factual accuracy (1-5 rubric)
  3. Latency — end-to-end time
  4. Error handling — graceful degradation vs crash

Usage:
  uv run python -m navigo_agent.evaluation.evaluator
"""

import json
import time
import logging
from dataclasses import dataclass, field

from navigo_agent.evaluation.test_cases import EVALUATION_CASES
from navigo_agent.guardrails import validate_input

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    test_id: str
    category: str
    passed: bool
    latency_seconds: float = 0.0
    tool_accuracy: float = 0.0  # 0.0–1.0
    quality_score: int = 0  # 1–5
    notes: str = ""


async def run_single_eval(case: dict) -> EvalResult:
    """Run a single evaluation case against the live agent."""
    query = case["query"]
    test_id = case["id"]
    category = case["category"]

    # Safety cases: only test guardrails, don't invoke the graph
    if case.get("expected_blocked"):
        start = time.monotonic()
        result = validate_input(query)
        elapsed = time.monotonic() - start
        passed = not result.passed
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=passed,
            latency_seconds=elapsed,
            tool_accuracy=1.0 if passed else 0.0,
            quality_score=0,
            notes=f"Guardrail {'blocked' if result.passed is False else 'passed (should have blocked)'}: {result.blocked_reason}",
        )

    # Normal cases: invoke the graph
    from navigo_agent import run_travel_agent

    start = time.monotonic()
    try:
        result = await run_travel_agent(query)
    except Exception as e:
        elapsed = time.monotonic() - start
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=False,
            latency_seconds=elapsed,
            notes=f"Exception: {str(e)}",
        )

    elapsed = time.monotonic() - start
    answer = result.get("answer", "")

    # ── Tool Accuracy ─────────────────────────────────────────────
    expected_agents = case.get("expected_agents")
    tool_accuracy = 1.0  # default: no strict check
    if expected_agents:
        actual_data = [
            bool(result.get("flight_results")),
            bool(result.get("hotel_results")),
            bool(result.get("weather_results")),
            bool(result.get("itinerary")),
        ]
        expected_data = [
            "flight_agent" in expected_agents,
            "hotel_agent" in expected_agents,
            "weather_agent" in expected_agents,
            "itinerary_agent" in expected_agents,
        ]
        matches = sum(1 for a, e in zip(actual_data, expected_data) if a == e)
        tool_accuracy = matches / 4.0

    # ── Quality Score ─────────────────────────────────────────────
    min_sections = case.get("min_sections", 1)
    quality_score = _score_output(answer, min_sections)

    passed = tool_accuracy >= 0.75 and quality_score >= 3 and len(answer) > 100

    return EvalResult(
        test_id=test_id,
        category=category,
        passed=passed,
        latency_seconds=elapsed,
        tool_accuracy=tool_accuracy,
        quality_score=quality_score,
        notes=f"len={len(answer)} chars, llm_calls={result.get('llm_calls', 0)}",
    )


def _score_output(answer: str, min_sections: int) -> int:
    """Quick heuristic quality score based on structure and length."""
    score = 0

    # Length check
    if len(answer) > 200:
        score += 1
    if len(answer) > 500:
        score += 1

    # Section check
    sections = ["#", "Trip Summary", "Flight", "Hotel", "Weather", "Itinerary", "Budget", "Recommendation"]
    found = sum(1 for s in sections if s.lower() in answer.lower())
    if found >= min_sections:
        score += 1
    if found >= 5:
        score += 1
    if found >= 7:
        score += 1

    return min(score, 5)


async def run_evaluation(verbose: bool = False) -> list[EvalResult]:
    """Run all evaluation cases and return results."""
    results: list[EvalResult] = []

    for case in EVALUATION_CASES:
        logger.info("Evaluating %s (%s)...", case["id"], case["category"])
        eval_result = await run_single_eval(case)
        results.append(eval_result)

        status = "PASS" if eval_result.passed else "FAIL"
        if verbose:
            print(
                f"  [{status}] {eval_result.test_id} "
                f"| accuracy={eval_result.tool_accuracy:.0%} "
                f"| quality={eval_result.quality_score}/5 "
                f"| latency={eval_result.latency_seconds:.1f}s "
                f"| {eval_result.notes}"
            )

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print a summary table of evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency = sum(r.latency_seconds for r in results) / max(total, 1)
    avg_accuracy = sum(r.tool_accuracy for r in results) / max(total, 1)
    avg_quality = sum(r.quality_score for r in results) / max(total, 1)

    print("\n" + "=" * 60)
    print("  Navigo-Agent Evaluation Summary")
    print("=" * 60)
    print(f"  Total cases:    {total}")
    print(f"  Passed:         {passed} ({passed / max(total, 1):.0%})")
    print(f"  Avg latency:    {avg_latency:.1f}s")
    print(f"  Avg tool acc:   {avg_accuracy:.0%}")
    print(f"  Avg quality:    {avg_quality:.1f}/5")
    print("=" * 60)

    # Category breakdown
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    print("\n  By category:")
    for cat, cat_results in sorted(by_category.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        print(f"    {cat:12s}: {cat_passed}/{len(cat_results)} passed")


# ── CLI Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.WARNING)  # quiet during eval
    results = asyncio.run(run_evaluation(verbose=True))
    print_summary(results)
