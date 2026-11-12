"""LLM-as-Judge evaluator for Navigo-Agent.

Measures:
  1. Intent — did the classifier pick the expected intent?
  2. Agent selection — did the right specialist agents run?
  3. Output quality — LLM-judged on relevance, completeness, faithfulness (1-5)
  4. Latency — end-to-end time
  5. Error handling — graceful degradation vs crash

Usage:
  uv run python -m navigo_agent.evaluation.evaluator
  uv run python -m navigo_agent.evaluation.evaluator --case weather_only_001
  uv run python -m navigo_agent.evaluation.evaluator --category full_trip --output eval.json
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from navigo_agent.config import get_llm
from navigo_agent.evaluation.test_cases import EVALUATION_CASES

logger = logging.getLogger(__name__)

ALL_AGENTS = ("flight_agent", "hotel_agent", "weather_agent", "itinerary_agent")


@dataclass
class EvalResult:
    test_id: str
    category: str
    passed: bool
    latency_seconds: float = 0.0
    tool_accuracy: float = 0.0  # 0.0–1.0
    intent: str = ""
    intent_correct: bool | None = None  # None → case has no expected intent
    agents: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    judge_failed: bool = False
    quality_score: int = 0  # 1–5
    quality_breakdown: dict = field(default_factory=dict)  # relevance, completeness, faithfulness
    notes: str = ""


# ── LLM-as-Judge Quality Scoring ──────────────────────────────────────

QUALITY_JUDGE_PROMPT = """You are an objective evaluator of a travel planning AI system. Score the assistant's response on three dimensions, each from 1 (poor) to 5 (excellent).

## User Query
{query}

## What a Complete Answer Covers
{scope}

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
- 5: Covers everything listed under "What a Complete Answer Covers". Do NOT penalise the answer for leaving out aspects the user did not ask about.
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


SCOPE_BY_INTENT = {
    "full_trip": "A full trip plan: flights, hotels, weather, a day-by-day itinerary, budget and practical tips.",
    "itinerary": "A day-by-day itinerary with practical tips.",
    "flight": "Flight options only (airlines, routes, rough prices). Hotels, weather and itinerary are not expected.",
    "hotel": "Hotel options only (names, areas, rough prices). Flights, weather and itinerary are not expected.",
    "weather": "The weather or forecast only. Flights, hotels and itinerary are not expected.",
}
DEFAULT_SCOPE = "A direct, useful answer to the question asked. A full trip plan is not expected."


def expected_scope(case: dict) -> str:
    """What the judge should treat as a complete answer for this case."""
    return SCOPE_BY_INTENT.get(case.get("expected_intent", ""), DEFAULT_SCOPE)


def agent_accuracy(expected: list[str] | None, actual: list[str]) -> float:
    """Share of the specialist agents whose ran/didn't-run matches the expectation."""
    if not expected:
        return 1.0
    matches = sum(1 for agent in ALL_AGENTS if (agent in expected) == (agent in actual))
    return matches / len(ALL_AGENTS)


async def _llm_quality_score(query: str, response: str, scope: str = DEFAULT_SCOPE) -> tuple[int, dict]:
    """Use LLM to score output quality on relevance, completeness, faithfulness.

    Returns (overall_score 1-5, breakdown_dict).
    """
    if len(response) < 50:
        return 1, {"relevance": 1, "completeness": 1, "faithfulness": 1, "summary": "Response too short to evaluate."}

    llm = get_llm(temperature=0.1)
    prompt = QUALITY_JUDGE_PROMPT.format(query=query, scope=scope, response=response[:6000])

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
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM quality scoring failed: %s — falling back to heuristic.", e)
        return _heuristic_score(response), {
            "relevance": 3, "completeness": 3, "faithfulness": 3,
            "summary": f"LLM judge failed: {e}",
            "judge_failed": True,
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
    """Run a single evaluation case against the live agent.

    A case with a `follow_up` query runs a second turn on the same
    conversation; the second answer is the one that is judged.
    """
    query = case["query"]
    test_id = case["id"]
    category = case["category"]
    follow_up = case.get("follow_up")

    from navigo_agent import run_travel_agent

    start = time.monotonic()
    try:
        result = await run_travel_agent(query, allow_clarification=False)
        first_intent = result.get("intent", "")
        first_agents = list(result.get("agents", []))
        llm_calls = result.get("llm_calls", 0)
        if follow_up:
            result = await run_travel_agent(
                follow_up, thread_id=result["thread_id"], allow_clarification=False
            )
            llm_calls += result.get("llm_calls", 0)
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - start
        return EvalResult(
            test_id=test_id,
            category=category,
            passed=False,
            latency_seconds=elapsed,
            notes=f"Exception: {e!s}",
        )

    elapsed = time.monotonic() - start
    answer = result.get("answer", "")
    errors = list(result.get("errors", []))

    # ── Intent & Agent Selection (always judged on the first turn) ─────
    expected_intent = case.get("expected_intent")
    intent_correct = (first_intent == expected_intent) if expected_intent else None
    tool_accuracy = agent_accuracy(case.get("expected_agents"), first_agents)

    # ── LLM-as-Judge Quality Score ─────────────────────────────────────
    judged_query = f"{query}\n(follow-up) {follow_up}" if follow_up else query
    scope = DEFAULT_SCOPE if follow_up else expected_scope(case)
    quality_score, quality_breakdown = await _llm_quality_score(judged_query, answer, scope)
    judge_failed = bool(quality_breakdown.pop("judge_failed", False))

    # ── Pass/Fail ──────────────────────────────────────────────────────
    notes = [f"len={len(answer)} chars", f"llm_calls={llm_calls}"]
    passed = (
        tool_accuracy >= 0.75
        and quality_score >= 3
        and len(answer) > 100
    )
    if follow_up and not result.get("follow_up"):
        passed = False
        notes.append("second turn was not treated as a follow-up")
    if intent_correct is False:
        notes.append(f"intent={first_intent!r} (expected {expected_intent!r})")
    if errors:
        notes.append(f"errors={len(errors)}")
    if judge_failed:
        notes.append("judge failed, heuristic score")

    return EvalResult(
        test_id=test_id,
        category=category,
        passed=passed,
        latency_seconds=elapsed,
        tool_accuracy=tool_accuracy,
        intent=first_intent,
        intent_correct=intent_correct,
        agents=first_agents,
        errors=errors,
        judge_failed=judge_failed,
        quality_score=quality_score,
        quality_breakdown=quality_breakdown,
        notes=", ".join(notes),
    )


# ── Batch Evaluation ──────────────────────────────────────────────────

def select_cases(
    case_ids: list[str] | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    """Filter the curated cases by id and/or category (no filter → all)."""
    known_ids = {case["id"] for case in EVALUATION_CASES}
    unknown = sorted(set(case_ids or []) - known_ids)
    if unknown:
        raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")
    return [
        case for case in EVALUATION_CASES
        if (not case_ids or case["id"] in case_ids)
        and (not categories or case["category"] in categories)
    ]


async def run_evaluation(
    verbose: bool = False,
    cases: list[dict] | None = None,
) -> list[EvalResult]:
    """Run the given evaluation cases (default: all) and return results."""
    results: list[EvalResult] = []

    for case in EVALUATION_CASES if cases is None else cases:
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


def write_report(results: list[EvalResult], path: str | Path) -> None:
    """Write the per-case results to a JSON file (to compare runs over time)."""
    passed = sum(1 for r in results if r.passed)
    report = {
        "total": len(results),
        "passed": passed,
        "results": [asdict(r) for r in results],
    }
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(results: list[EvalResult]) -> None:
    """Print a summary table of evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    intent_checked = [r for r in results if r.intent_correct is not None]
    intent_ok = sum(1 for r in intent_checked if r.intent_correct)
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
    print(f"  Intent accuracy:    {intent_ok}/{len(intent_checked)}")
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

def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description="Run the Navigo-Agent evaluation suite.")
    parser.add_argument("--case", action="append", help="Run only this case id (repeatable).")
    parser.add_argument("--category", action="append", help="Run only this category (repeatable).")
    parser.add_argument("--output", help="Write per-case results to this JSON file.")
    args = parser.parse_args(argv)

    try:
        cases = select_cases(args.case, args.category)
    except ValueError as e:
        parser.error(str(e))
    if not cases:
        parser.error("No evaluation cases match the given filters.")

    logging.basicConfig(level=logging.WARNING)  # quiet during eval
    results = asyncio.run(run_evaluation(verbose=True, cases=cases))
    print_summary(results)
    if args.output:
        write_report(results, args.output)
        print(f"\n  Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
