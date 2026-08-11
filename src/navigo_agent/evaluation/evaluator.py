"""LLM-as-Judge evaluator for Navigo-Agent.

Measures:
  1. Tool selection accuracy — did the supervisor dispatch the right agents?
  2. Output quality — LLM-judged on relevance, completeness, faithfulness (1-5)
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
from navigo_agent.config import get_llm

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    test_id: str
    category: str
    passed: bool
    latency_seconds: float = 0.0
    tool_accuracy: float = 0.0  # 0.0–1.0
    quality_score: int = 0  # 1–5
    quality_breakdown: dict = field(default_factory=dict)  # relevance, completeness, faithfulness
    notes: str = ""


# ── LLM-as-Judge Quality Scoring ──────────────────────────────────────

QUALITY_JUDGE_PROMPT = """You are an objective evaluator of a travel planning AI system. Score the assistant's response on three dimensions, each from 1 (poor) to 5 (excellent).

## User Query
{query}

## Assistant Response
{response}

## Scoring Rubric

### Relevance (1-5)
- 5: Every section directly addresses the user's query. No irrelevant or tangential content.
- 4: Mostly relevant, minor tangents that don't detract from usefulness.
- 3: Some relevant content mixed with general/off-topic information.
- 2: Mostly generic content, barely addresses the specific query.
- 1: Completely off-topic or irrelevant.

### Completeness (1-5)
- 5: Covers all requested aspects (flights, hotels, weather, itinerary, budget, tips). No missing sections the user asked for.
- 4: Covers most aspects but missing 1-2 minor details.
- 3: Covers some aspects but missing significant sections the user would expect.
- 2: Very sparse, only addresses one aspect.
- 1: Almost no useful information provided.

### Faithfulness (1-5)
- 5: No factual hallucinations. All claims seem verifiable. Appropriately marks uncertain info as "N/A" or "check official".
- 4: Minor unverifiable claims but nothing clearly fabricated.
- 3: Some claims that seem exaggerated or unverifiable.
- 2: Multiple clearly fabricated facts or prices.
- 1: Heavily hallucinated, full of made-up information.

## Output Format

Respond with ONLY a JSON object:
{{"relevance": <int 1-5>, "completeness": <int 1-5>, "faithfulness": <int 1-5>, "summary": "<one sentence>"}}
"""


async def _llm_quality_score(query: str, response: str) -> tuple[int, dict]:
    """Use LLM to score output quality on relevance, completeness, faithfulness.

    Returns (overall_score 1-5, breakdown_dict).
    """
    if len(response) < 50:
        return 1, {"relevance": 1, "completeness": 1, "faithfulness": 1, "summary": "Response too short to evaluate."}

    llm = get_llm(temperature=0.1)
    prompt = QUALITY_JUDGE_PROMPT.format(query=query, response=response[:3000])

    try:
        result = await llm.ainvoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)

        # Parse JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        data = json.loads(text)

        scores = {
            "relevance": int(data.get("relevance", 3)),
            "completeness": int(data.get("completeness", 3)),
            "faithfulness": int(data.get("faithfulness", 3)),
            "summary": data.get("summary", ""),
        }
        # Overall = average of three dimensions, rounded
        overall = round((scores["relevance"] + scores["completeness"] + scores["faithfulness"]) / 3)
        return overall, scores
    except Exception as e:
        logger.warning("LLM quality scoring failed: %s — falling back to heuristic.", e)
        return _heuristic_score(response), {
            "relevance": 3, "completeness": 3, "faithfulness": 3,
            "summary": f"LLM judge failed: {e}",
        }


def _heuristic_score(answer: str) -> int:
    """Fallback heuristic quality score based on structure and length."""
    score = 1
    if len(answer) > 200:
        score += 1
    if len(answer) > 500:
        score += 1
    sections = ["#", "Trip Summary", "Flight", "Hotel", "Weather", "Itinerary", "Budget", "Recommendation"]
    found = sum(1 for s in sections if s.lower() in answer.lower())
    if found >= 3:
        score += 1
    if found >= 5:
        score += 1
    return min(score, 5)


# ── Single Case Evaluation ────────────────────────────────────────────

async def run_single_eval(case: dict) -> EvalResult:
    """Run a single evaluation case against the live agent."""
    query = case["query"]
    test_id = case["id"]
    category = case["category"]

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

    # ── Tool Accuracy ─────────────────────────────────────────────────
    expected_agents = case.get("expected_agents")
    tool_accuracy = 1.0
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

    # ── LLM-as-Judge Quality Score ─────────────────────────────────────
    quality_score, quality_breakdown = await _llm_quality_score(query, answer)

    # ── Pass/Fail ──────────────────────────────────────────────────────
    passed = (
        tool_accuracy >= 0.75
        and quality_score >= 3
        and len(answer) > 100
    )

    return EvalResult(
        test_id=test_id,
        category=category,
        passed=passed,
        latency_seconds=elapsed,
        tool_accuracy=tool_accuracy,
        quality_score=quality_score,
        quality_breakdown=quality_breakdown,
        notes=f"len={len(answer)} chars, llm_calls={result.get('llm_calls', 0)}",
    )


# ── Batch Evaluation ──────────────────────────────────────────────────

async def run_evaluation(verbose: bool = False) -> list[EvalResult]:
    """Run all evaluation cases and return results."""
    results: list[EvalResult] = []

    for case in EVALUATION_CASES:
        logger.info("Evaluating %s (%s)...", case["id"], case["category"])
        eval_result = await run_single_eval(case)
        results.append(eval_result)

        if verbose:
            breakdown = eval_result.quality_breakdown
            print(
                f"  [{'PASS' if eval_result.passed else 'FAIL'}] {eval_result.test_id} "
                f"| accuracy={eval_result.tool_accuracy:.0%} "
                f"| quality={eval_result.quality_score}/5 "
                f"(R:{breakdown.get('relevance','?')} C:{breakdown.get('completeness','?')} F:{breakdown.get('faithfulness','?')}) "
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
    avg_relevance = sum(r.quality_breakdown.get("relevance", 0) for r in results) / max(total, 1)
    avg_completeness = sum(r.quality_breakdown.get("completeness", 0) for r in results) / max(total, 1)
    avg_faithfulness = sum(r.quality_breakdown.get("faithfulness", 0) for r in results) / max(total, 1)

    print("\n" + "=" * 60)
    print("  Navigo-Agent Evaluation Summary")
    print("=" * 60)
    print(f"  Total cases:        {total}")
    print(f"  Passed:             {passed} ({passed / max(total, 1):.0%})")
    print(f"  Avg latency:        {avg_latency:.1f}s")
    print(f"  Avg tool accuracy:  {avg_accuracy:.0%}")
    print(f"  Avg quality:        {avg_quality:.1f}/5")
    print(f"    - Relevance:      {avg_relevance:.1f}/5")
    print(f"    - Completeness:   {avg_completeness:.1f}/5")
    print(f"    - Faithfulness:   {avg_faithfulness:.1f}/5")
    print("=" * 60)

    # Category breakdown
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    print("\n  By category:")
    for cat, cat_results in sorted(by_category.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_quality = sum(r.quality_score for r in cat_results) / max(len(cat_results), 1)
        print(f"    {cat:12s}: {cat_passed}/{len(cat_results)} passed | avg quality {cat_quality:.1f}/5")

    # Individual results table
    print("\n  Details:")
    print(f"  {'ID':<25s} {'Category':<12s} {'Pass':>5s} {'Acc':>5s} {'Qual':>5s} {'Lat':>6s}")
    print(f"  {'-'*25} {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    for r in results:
        print(f"  {r.test_id:<25s} {r.category:<12s} {'PASS' if r.passed else 'FAIL':>5s} {r.tool_accuracy:>4.0%} {r.quality_score:>4d}/5 {r.latency_seconds:>5.1f}s")


# ── CLI Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.WARNING)  # quiet during eval
    results = asyncio.run(run_evaluation(verbose=True))
    print_summary(results)
