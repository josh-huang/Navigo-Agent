"""Shared ReAct loop for the Tavily-search agents — native tool calling.

The LLM is given a single `web_search` tool via `bind_tools()`. Each turn it
either emits tool calls (→ we run the searches and feed the results back as
ToolMessages) or answers in plain markdown (→ that is the final answer).

This replaces the earlier hand-rolled protocol where the model had to wrap
both its actions and its whole markdown answer inside a JSON string, which
broke whenever the model mis-escaped a quote or newline.

Each agent only needs to supply:
  - A system prompt + initial HumanMessage
  - An async search function (query -> text)
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 3
MAX_OBSERVATION_CHARS = 3000
SEARCH_TIMEOUT_SECONDS = 30  # a hung search must not stall the whole request

SEARCH_TOOL_NAME = "web_search"

# OpenAI-style function schema — accepted by ChatGroq.bind_tools()
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the web for real-time travel information (flights, airlines, hotels, "
            "neighbourhoods, attractions, transport, prices). Returns text snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, self-contained search query.",
                },
            },
            "required": ["query"],
        },
    },
}

FORCE_FINAL_PROMPT = (
    "You have used all your searches. Write your final answer now, in markdown, "
    "using only the information gathered above. Do not call any tools."
)
FALLBACK_ANSWER = "Information could not be retrieved. Please try a more specific query."


def _text(response) -> str:
    """Plain text of an AIMessage (content may be a list of blocks)."""
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content or "").strip()


async def _run_search(
    call: dict,
    search_fn: Callable[[str], Awaitable[str]],
    agent_name: str,
    max_observation_chars: int,
    errors: list[str],
) -> ToolMessage:
    """Execute one tool call. Always returns a ToolMessage — never raises."""
    call_id = call.get("id") or ""
    args = call.get("args") or {}
    query = str(args.get("query") or "").strip() if isinstance(args, dict) else ""

    if call.get("name") != SEARCH_TOOL_NAME:
        observation = f"Unknown tool '{call.get('name')}'. The only tool is {SEARCH_TOOL_NAME}."
    elif not query:
        observation = "Search failed: the 'query' argument was empty."
    else:
        try:
            observation = await asyncio.wait_for(search_fn(query), timeout=SEARCH_TIMEOUT_SECONDS)
            if len(observation) > max_observation_chars:
                observation = observation[:max_observation_chars] + "\n... [truncated]"
            logger.info("%s: search '%s' → %d chars", agent_name, query[:60], len(observation))
        except TimeoutError:
            observation = f"Search failed: no response within {SEARCH_TIMEOUT_SECONDS}s."
            logger.warning("%s: search '%s' timed out", agent_name, query[:60])
            errors.append(f"{agent_name}(search): timed out after {SEARCH_TIMEOUT_SECONDS}s")
        except Exception as e:  # noqa: BLE001
            observation = f"Search failed: {e}"
            logger.warning("%s: search failed: %s", agent_name, e)
            errors.append(f"{agent_name}(search): {e}")

    return ToolMessage(content=observation, tool_call_id=call_id, name=SEARCH_TOOL_NAME)


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
    """Tool-calling ReAct loop: reason → web_search → observe → ... → final answer.

    Args:
        llm: Chat model supporting `bind_tools()` (ChatGroq).
        conversation: Initial message list (SystemMessage + HumanMessage). Mutated.
        search_fn: Async callable: query string → result text.
        agent_name: Used in log / error messages.
        max_iterations: Max search rounds before a final answer is forced.
        max_observation_chars: Truncate each search result to this length.
        llm_calls: Starting count (default 0 → the return value is this loop's own calls).

    Returns:
        (final_answer: str, total_llm_calls: int, errors: list[str])
    """
    errors: list[str] = []
    tool_llm = llm.bind_tools([WEB_SEARCH_TOOL])
    # Tools stay declared (the history contains tool messages) but may not be called.
    final_llm = llm.bind_tools([WEB_SEARCH_TOOL], tool_choice="none")

    final_answer = ""
    need_forced_final = True

    for iteration in range(1, max_iterations + 1):
        logger.info("%s ReAct round %d/%d", agent_name, iteration, max_iterations)

        try:
            response = await tool_llm.ainvoke(conversation)
            llm_calls += 1
        except Exception as e:  # noqa: BLE001
            # e.g. the provider rejecting a malformed tool call — fall through
            # to a tool-free final answer with whatever has been gathered.
            logger.error("%s LLM call failed (round %d): %s", agent_name, iteration, e)
            errors.append(f"{agent_name}(llm,iter={iteration}): {e}")
            break

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_answer = _text(response)
            need_forced_final = not final_answer
            if final_answer:
                logger.info("%s: final answer at round %d", agent_name, iteration)
            break

        # Every tool call must be answered, so run them all (concurrently).
        conversation.append(response)
        conversation.extend(await asyncio.gather(*(
            _run_search(call, search_fn, agent_name, max_observation_chars, errors)
            for call in tool_calls
        )))

    if need_forced_final:
        logger.info("%s: forcing final answer without tools.", agent_name)
        conversation.append(HumanMessage(content=FORCE_FINAL_PROMPT))
        try:
            response = await final_llm.ainvoke(conversation)
            llm_calls += 1
            final_answer = _text(response)
        except Exception as e:  # noqa: BLE001
            logger.error("%s: forced final answer failed: %s", agent_name, e)
            errors.append(f"{agent_name}(force_final): {e}")

    return final_answer or FALLBACK_ANSWER, llm_calls, errors
