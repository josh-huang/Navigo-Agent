"""Fake chat models that stand in for the orchestrator's LLM."""

import json
import re
import uuid

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


def tool_call(name: str, **args) -> dict:
    """A tool call as the orchestrator's LLM would emit it (= one delegation)."""
    return {"name": name, "args": args, "id": f"call_{name}_{uuid.uuid4().hex[:6]}", "type": "tool_call"}


class _BoundLLM:
    def __init__(self, llm, tool_choice):
        self._llm, self._tool_choice = llm, tool_choice

    async def ainvoke(self, messages):
        return await self._llm.respond(messages, self._tool_choice)


class ScriptedOrchestratorLLM:
    """Stands in for the orchestrator's chat model.

    Each positional argument is one delegation turn: the sub-agents to call.
    Once the script is used up (or tools are switched off) it answers.

    `calls` records every invocation as (messages, tool_choice).
    """

    def __init__(self, *rounds: list[str], answer: str = "FINAL PLAN"):
        self.rounds = list(rounds)
        self.answer = answer
        self.calls: list[tuple[list, str | None]] = []

    def bind_tools(self, _tools, tool_choice=None):
        return _BoundLLM(self, tool_choice)

    async def respond(self, messages, tool_choice):
        self.calls.append((list(messages), tool_choice))
        if tool_choice != "none" and self.rounds:
            return AIMessage(content="", tool_calls=[tool_call(name, task=f"brief for {name}") for name in self.rounds.pop(0)])
        return AIMessage(content=self.answer)


class SensibleOrchestratorLLM:
    """Plans like a well-behaved orchestrator, from the request brief it is given:
    research first, then the itinerary for trip intents, then the answer.
    """

    def __init__(self, prompts: list[str], answer_prompts: list[str]):
        self.prompts, self.answer_prompts = prompts, answer_prompts

    def bind_tools(self, _tools, tool_choice=None):
        return _BoundLLM(self, tool_choice)

    async def respond(self, messages, tool_choice):
        brief = messages[1].content
        self.prompts.append(brief)
        if tool_choice != "none":
            if "Reports received from: none yet" in brief:
                return AIMessage(content="", tool_calls=[
                    tool_call("flight_agent", task="find flights"),
                    tool_call("hotel_agent", task="find hotels"),
                    tool_call("weather_agent", city="Tokyo"),
                ])
            is_trip = "- Intent: full_trip" in brief or "- Intent: itinerary" in brief
            if is_trip and "itinerary_agent" in brief.split("Sub-agents not used yet:")[1].split("\n")[0]:
                return AIMessage(content="", tool_calls=[tool_call("itinerary_agent", task="build the plan")])
        self.answer_prompts.append(brief)
        count = len(self.answer_prompts)
        return AIMessage(content=f"FINAL PLAN #{count}" if count > 1 else "FINAL PLAN")


class StreamingOrchestratorLLM(GenericFakeChatModel):
    """A real chat model (streams tokens through LangChain callbacks) fed scripted messages.

    GenericFakeChatModel cannot stream a tool-call-only message, so that case is added here.
    """

    def bind_tools(self, _tools, **_kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        if message.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ]))
            return
        for token in re.split(r"(\s)", message.content):
            if token:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
