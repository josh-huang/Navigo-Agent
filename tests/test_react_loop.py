"""Tool-calling ReAct loop tests (no network — scripted fake chat model)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from navigo_agent.graph.agents.react_utils import (
    FALLBACK_ANSWER,
    SEARCH_TOOL_NAME,
    run_react_loop,
)


def _search_call(query: str, call_id: str = "call_1", name: str = SEARCH_TOOL_NAME) -> dict:
    return {"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}


class FakeToolLLM:
    """Scripted chat model. Records every bind_tools() / ainvoke() call."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []  # {"tool_choice": ..., "messages": [...]}
        self.bound: list[dict] = []

    def bind_tools(self, tools, tool_choice=None):
        self.bound.append({"tools": tools, "tool_choice": tool_choice})
        return _Bound(self, tool_choice)


class _Bound:
    def __init__(self, parent: FakeToolLLM, tool_choice):
        self.parent, self.tool_choice = parent, tool_choice

    async def ainvoke(self, messages):
        self.parent.calls.append({"tool_choice": self.tool_choice, "messages": list(messages)})
        step = self.parent.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _conversation() -> list:
    return [SystemMessage(content="system"), HumanMessage(content="Find flights to Tokyo")]


async def _run(llm, search, **kwargs):
    conversation = _conversation()
    result = await run_react_loop(
        llm=llm, conversation=conversation, search_fn=search, agent_name="flight_agent", **kwargs,
    )
    return result, conversation


@pytest.mark.asyncio
async def test_search_then_final_answer():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[_search_call("Dhaka to Tokyo flights")]),
        AIMessage(content="## Flight Options\n\nANA, JAL"),
    ])
    queries: list[str] = []

    async def search(query):
        queries.append(query)
        return "ANA and JAL fly this route"

    (answer, llm_calls, errors), conversation = await _run(llm, search)

    assert answer == "## Flight Options\n\nANA, JAL"
    assert llm_calls == 2
    assert errors == []
    assert queries == ["Dhaka to Tokyo flights"]
    # The search tool was actually offered to the model
    assert llm.bound[0]["tools"][0]["function"]["name"] == SEARCH_TOOL_NAME
    # Tool result fed back with the matching call id
    tool_msg = conversation[-1]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == "ANA and JAL fly this route"


@pytest.mark.asyncio
async def test_direct_answer_without_search():
    llm = FakeToolLLM([AIMessage(content="Markdown with \"quotes\" and\nnewlines — no JSON escaping needed")])

    async def search(_query):
        raise AssertionError("search must not be called")

    (answer, llm_calls, errors), _ = await _run(llm, search)
    assert answer.startswith('Markdown with "quotes"')
    assert llm_calls == 1
    assert errors == []


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_turn_are_all_answered():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[_search_call("q1", "a"), _search_call("q2", "b")]),
        AIMessage(content="done"),
    ])

    async def search(query):
        return f"result for {query}"

    (answer, _, _), conversation = await _run(llm, search)
    assert answer == "done"
    tool_msgs = [m for m in conversation if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_msgs] == [
        ("a", "result for q1"), ("b", "result for q2"),
    ]


@pytest.mark.asyncio
async def test_search_budget_forces_tool_free_final_answer():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[_search_call("q1", "1")]),
        AIMessage(content="", tool_calls=[_search_call("q2", "2")]),
        AIMessage(content="final from gathered data"),
    ])
    count = 0

    async def search(_query):
        nonlocal count
        count += 1
        return "data"

    (answer, llm_calls, errors), _ = await _run(llm, search, max_iterations=2)
    assert answer == "final from gathered data"
    assert count == 2
    assert llm_calls == 3
    assert errors == []
    # Last call is made with tools disabled, after an explicit instruction
    assert [c["tool_choice"] for c in llm.calls] == [None, None, "none"]
    assert "final answer" in llm.calls[-1]["messages"][-1].content.lower()


@pytest.mark.asyncio
async def test_search_failure_is_reported_to_the_model():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[_search_call("q1")]),
        AIMessage(content="best effort answer"),
    ])

    async def search(_query):
        raise RuntimeError("tavily down")

    (answer, _, errors), conversation = await _run(llm, search)
    assert answer == "best effort answer"
    assert errors == ["flight_agent(search): tavily down"]
    assert "Search failed: tavily down" in conversation[-1].content


@pytest.mark.asyncio
async def test_long_observation_is_truncated():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[_search_call("q1")]),
        AIMessage(content="ok"),
    ])

    async def search(_query):
        return "x" * 500

    (_, _, _), conversation = await _run(llm, search, max_observation_chars=100)
    assert conversation[-1].content == "x" * 100 + "\n... [truncated]"


@pytest.mark.asyncio
async def test_bad_tool_calls_do_not_crash():
    llm = FakeToolLLM([
        AIMessage(content="", tool_calls=[
            _search_call("q", "1", name="book_flight"),
            {"name": SEARCH_TOOL_NAME, "args": {}, "id": "2", "type": "tool_call"},
        ]),
        AIMessage(content="answer"),
    ])

    async def search(_query):
        raise AssertionError("search must not be called")

    (answer, _, errors), conversation = await _run(llm, search)
    assert answer == "answer"
    assert errors == []
    tool_msgs = [m for m in conversation if isinstance(m, ToolMessage)]
    assert "Unknown tool 'book_flight'" in tool_msgs[0].content
    assert "empty" in tool_msgs[1].content


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_tool_free_answer():
    """e.g. the provider rejecting a malformed tool call."""
    llm = FakeToolLLM([
        RuntimeError("tool_use_failed"),
        AIMessage(content="answer without searching"),
    ])

    async def search(_query):
        return "unused"

    (answer, llm_calls, errors), _ = await _run(llm, search)
    assert answer == "answer without searching"
    assert llm_calls == 1  # only the successful call is counted
    assert errors == ["flight_agent(llm,iter=1): tool_use_failed"]
    assert llm.calls[-1]["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_total_failure_returns_fallback_text():
    llm = FakeToolLLM([RuntimeError("down"), RuntimeError("still down")])

    async def search(_query):
        return "unused"

    (answer, llm_calls, errors), _ = await _run(llm, search)
    assert answer == FALLBACK_ANSWER
    assert llm_calls == 0
    assert len(errors) == 2


@pytest.mark.asyncio
async def test_empty_answer_triggers_forced_final():
    llm = FakeToolLLM([AIMessage(content="   "), AIMessage(content="real answer")])

    async def search(_query):
        return "unused"

    (answer, llm_calls, _), _ = await _run(llm, search)
    assert answer == "real answer"
    assert llm_calls == 2
