"""Shared ReAct loop utilities for all Tavily-search-based agents.

Extracted from flight.py — the same ReActDecision model, JSON parser,
and observation loop are reused by flight_agent, hotel_agent, and
itinerary_agent.

Each agent only needs to supply:
  - A system prompt builder function
  - An initial HumanMessage
  - Temperature and agent-specific config

Usage:
    from navigo_agent.graph.agents.react_utils import (
        ReActDecision, parse_react_response, run_react_loop,
    )
"""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 3
MAX_OBSERVATION_CHARS = 3000


# ── ReAct Decision Model ──────────────────────────────────────────────

class ReActDecision(BaseModel):
    """Single step in the ReAct loop — parsed from LLM JSON output."""
    reasoning: str = Field(description="What info do I still need, and why?")
    action: Literal["search", "final_answer"] = Field(
        description="'search' to run a web search, 'final_answer' to produce the result"
    )
    query: str | None = Field(
        default=None,
        description="Targeted search query (required when action='search')",
    )
    content: str | None = Field(
        default=None,
        description="Final formatted content (required when action='final_answer')",
    )


# ── JSON Parsing (defensive, 3-level fallback) ────────────────────────

def parse_react_response(text: str) -> ReActDecision:
    """Parse LLM output into ReActDecision. Falls back to final_answer on failure.

    Strategy:
      1. Strip markdown code fences, try direct JSON parse
      2. Regex-extract JSON object, try parse
      3. Fallback: treat entire response as final_answer
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Attempt 1: direct JSON parse
    try:
        return ReActDecision(**json.loads(cleaned))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug("ReAct JSON parse attempt 1 failed: %s", e)

    # Attempt 2: find JSON object via regex
    for pattern in [
        r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}',
        r'\{.*\}',
    ]:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return ReActDecision(**json.loads(match.group()))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.debug("ReAct JSON parse attempt 2 failed: %s", e)

    # Fallback: treat entire response as final answer
    logger.warning("Failed to parse ReAct JSON. Treating response as final answer.")
    return ReActDecision(
        reasoning="Parsing failed, defaulting to final answer.",
        action="final_answer",
        content=text[:3000],
    )


# ── ReAct Loop Runner ─────────────────────────────────────────────────

async def run_react_loop(
    *,
    llm,
    conversation: list,
    search_fn: Callable[[str], Awaitable[str]],
    agent_name: str = "agent",
    max_iterations: int = MAX_REACT_ITERATIONS,
    max_observation_chars: int = MAX_OBSERVATION_CHARS,
    llm_calls: int = 0,
) -> tuple[str, int, list[str]]:
    """Run a generic ReAct loop: reason → search → observe → repeat → final answer.

    Args:
        llm: Configured ChatGroq instance.
        conversation: Initial message list (SystemMessage + HumanMessage).
        search_fn: Async callable that takes a query string and returns result text
                   (already unwrapped from MCP format).
        agent_name: Used in log messages for traceability.
        max_iterations: Max ReAct iterations before forcing final answer.
        max_observation_chars: Truncate each observation to this length.
        llm_calls: Starting LLM call count (accumulated).

    Returns:
        (final_answer: str, total_llm_calls: int, errors: list[str])
    """
    final_answer = ""
    errors: list[str] = []

    for iteration in range(1, max_iterations + 1):
        logger.info("%s ReAct iteration %d/%d", agent_name, iteration, max_iterations)

        # --- Reason + Act ---
        try:
            response = await llm.ainvoke(conversation)
            llm_calls += 1
        except Exception as e:  # noqa: BLE001
            logger.error("%s LLM call failed (iter %d): %s", agent_name, iteration, e)
            errors.append(f"{agent_name}(llm,iter={iteration}): {e}")
            break

        decision = parse_react_response(response.content)

        # --- Final Answer ---
        if decision.action == "final_answer" and decision.content:
            final_answer = decision.content.strip()
            logger.info("%s: final answer at iteration %d", agent_name, iteration)
            break

        # --- Search ---
        if decision.action == "search" and decision.query:
            try:
                observation = await search_fn(decision.query)
                if len(observation) > max_observation_chars:
                    observation = observation[:max_observation_chars] + "\n... [truncated]"
                logger.info("%s: search '%s' → %d chars", agent_name, decision.query[:60], len(observation))
            except Exception as e:  # noqa: BLE001
                observation = f"Search failed: {e}"
                logger.warning("%s: search failed (iter %d): %s", agent_name, iteration, e)
                errors.append(f"{agent_name}(search,iter={iteration}): {e}")

            conversation.append(AIMessage(content=response.content))
            conversation.append(HumanMessage(
                content=f"[Search results for: {decision.query}]\n\n{observation}\n\n"
                        f"What specific information are you still missing? "
                        f"If you have enough data, provide your final answer. "
                        f"Otherwise, search for the missing piece with a refined query."
            ))
            continue

        # --- Unparseable → nudge the LLM ---
        logger.warning("%s: unparseable response at iteration %d", agent_name, iteration)
        conversation.append(AIMessage(content=response.content))
        conversation.append(HumanMessage(
            content='I could not parse your response. Reply with ONLY a JSON object:\n'
                    '{"action": "search", "query": "<your search query>", "reasoning": "..."}\n'
                    'or\n'
                    '{"action": "final_answer", "content": "Your answer...", "reasoning": "..."}'
        ))

    # ── Force final answer if loop exhausted ───────────────────────────
    if not final_answer:
        logger.warning("%s: max ReAct iterations reached. Forcing final answer.", agent_name)
        try:
            conversation.append(HumanMessage(
                content="You have reached the maximum number of searches. "
                        "Produce your best analysis now with the data you have. "
                        'Respond with {"action": "final_answer", "content": "..."}'
            ))
            response = await llm.ainvoke(conversation)
            llm_calls += 1
            decision = parse_react_response(response.content)
            final_answer = decision.content or response.content or "Analysis could not be completed."
        except Exception as e:  # noqa: BLE001
            logger.error("%s: forced final answer failed: %s", agent_name, e)
            final_answer = "Information could not be retrieved. Please try a more specific query."
            errors.append(f"{agent_name}(force_final): {e}")

    return final_answer, llm_calls, errors
