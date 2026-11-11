"""Multi-turn conversation tests: follow-ups carry history and trip slots forward."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import navigo_agent
import navigo_agent.graph.routing as routing_mod
import navigo_agent.graph.slots as slots_mod
from navigo_agent import streaming
from navigo_agent.graph.slots import (
    MAX_HISTORY_TURNS,
    MAX_STORED_ANSWER_CHARS,
    TripSlots,
    append_turn,
    conversation_block,
    extract_follow_up,
    format_history,
)
from navigo_agent.session import plan_run
from navigo_agent.state import build_initial_state

TOKYO = TripSlots(origin="Dhaka", destination="Tokyo", duration_days=5, budget="mid-range")
HISTORY = [{"user": "Plan a 5-day trip to Tokyo from Dhaka", "assistant": "## Trip Summary\nANA flights, Shinjuku hotel"}]


class TestHistoryHelpers:
    def test_append_turn_trims_and_truncates(self):
        history = []
        for i in range(MAX_HISTORY_TURNS + 3):
            history = append_turn(history, f" q{i} ", "a" * (MAX_STORED_ANSWER_CHARS + 500))
        assert len(history) == MAX_HISTORY_TURNS
        assert history[-1]["user"] == f"q{MAX_HISTORY_TURNS + 2}"
        assert history[0]["user"] == "q3"  # oldest turns dropped
        assert len(history[-1]["assistant"]) == MAX_STORED_ANSWER_CHARS

    def test_format_history(self):
        text = format_history([{"user": "hi", "assistant": "x" * 50}], answer_chars=10)
        assert text == "User: hi\nAssistant: " + "x" * 10 + " …"
        assert format_history([]) == ""
        assert format_history(None) == ""

    def test_conversation_block_empty_without_history(self):
        assert conversation_block([]) == ""
        block = conversation_block(HISTORY)
        assert "Conversation So Far" in block
        assert "Plan a 5-day trip to Tokyo from Dhaka" in block


class TestExtractFollowUp:
    async def _run(self, content):
        llm = AsyncMock()
        if isinstance(content, Exception):
            llm.ainvoke.side_effect = content
        else:
            llm.ainvoke.return_value = AIMessage(content=content)
        with patch.object(slots_mod, "get_llm", lambda **_: llm):
            result = await extract_follow_up("make it cheaper", HISTORY, TOKYO.model_dump())
        return result, llm

    @pytest.mark.asyncio
    async def test_prompt_contains_previous_trip_and_history(self):
        payload = {"intent": "full_trip", "budget": "budget", "is_follow_up": True}
        (intent, slots, is_follow_up), llm = await self._run(json.dumps(payload))
        system, human = llm.ainvoke.await_args.args[0]
        assert '"destination": "Tokyo"' in system.content
        assert "Plan a 5-day trip to Tokyo from Dhaka" in system.content
        assert '"is_follow_up"' in system.content
        assert human.content == "make it cheaper"
        assert (intent, slots.budget, is_follow_up) == ("full_trip", "budget", True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag,expected", [(False, False), ("false", False), ("no", False), (True, True), ("true", True)])
    async def test_is_follow_up_flag_parsing(self, flag, expected):
        (_, _, is_follow_up), _ = await self._run(json.dumps({"intent": "weather", "is_follow_up": flag}))
        assert is_follow_up is expected

    @pytest.mark.asyncio
    async def test_missing_flag_defaults_to_follow_up(self):
        (_, _, is_follow_up), _ = await self._run(json.dumps({"intent": "hotel"}))
        assert is_follow_up is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content", ["not json", RuntimeError("rate limited")])
    async def test_failure_keeps_the_conversation(self, content):
        (intent, slots, is_follow_up), _ = await self._run(content)
        assert (intent, slots, is_follow_up) == (None, TripSlots(), True)


class TestIntentClassifierFollowUp:
    def _state(self, query):
        return build_initial_state(query, history=HISTORY, previous_trip=TOKYO.model_dump())

    @pytest.mark.asyncio
    async def test_follow_up_merges_slots_and_prefers_llm_intent(self):
        # Regex alone would say "hotel"; with context the LLM asks for a full re-plan.
        follow = AsyncMock(return_value=("full_trip", TripSlots(budget="budget"), True))
        with patch.object(routing_mod, "extract_follow_up", follow):
            update = await routing_mod.intent_classifier(self._state("make it cheaper with budget hotels"))
        assert update["user_intent"] == "full_trip"
        assert update["trip"]["destination"] == "Tokyo"   # carried over
        assert update["trip"]["origin"] == "Dhaka"
        assert update["trip"]["duration_days"] == 5
        assert update["trip"]["budget"] == "budget"       # changed by the follow-up
        assert update["history"] == HISTORY
        assert update["extracted_destination"] == "Tokyo"

    @pytest.mark.asyncio
    async def test_new_topic_drops_history_and_previous_slots(self):
        follow = AsyncMock(return_value=("full_trip", TripSlots(destination="Dubai"), False))
        with patch.object(routing_mod, "extract_follow_up", follow):
            update = await routing_mod.intent_classifier(self._state("Plan a trip to Dubai"))
        assert update["history"] == []
        assert update["trip"]["destination"] == "Dubai"
        assert update["trip"]["origin"] is None           # NOT inherited from Tokyo
        assert update["trip"]["duration_days"] is None

    @pytest.mark.asyncio
    async def test_first_turn_does_not_use_follow_up_extractor(self):
        follow = AsyncMock()
        extract = AsyncMock(return_value=("weather", TripSlots(destination="Tokyo")))
        with (
            patch.object(routing_mod, "extract_follow_up", follow),
            patch.object(routing_mod, "extract_trip_request", extract),
        ):
            update = await routing_mod.intent_classifier(build_initial_state("Weather in Tokyo?"))
        follow.assert_not_awaited()
        assert update["history"] == []


class TestPlanRun:
    def _graph(self, snapshot):
        return SimpleNamespace(aget_state=AsyncMock(return_value=snapshot))

    @pytest.mark.asyncio
    async def test_finished_thread_seeds_follow_up(self):
        snapshot = SimpleNamespace(next=(), interrupts=(), values={
            "user_query": "Plan a 5-day trip to Tokyo from Dhaka",
            "messages": [HumanMessage(content="q"), AIMessage(content="## Plan")],
            "history": [{"user": "earlier", "assistant": "earlier answer"}],
            "trip": TOKYO.model_dump(),
        })
        plan = await plan_run(self._graph(snapshot), "make it cheaper", "user_prev")
        assert plan.resumed is False
        assert plan.thread_id != "user_prev"
        state = plan.graph_input
        assert state["user_query"] == "make it cheaper"
        assert state["history"] == [
            {"user": "earlier", "assistant": "earlier answer"},
            {"user": "Plan a 5-day trip to Tokyo from Dhaka", "assistant": "## Plan"},
        ]
        assert state["previous_trip"]["destination"] == "Tokyo"
        assert state["trip"] == {}  # decided by the intent node, not copied blindly

    @pytest.mark.asyncio
    @pytest.mark.parametrize("values", [{}, {"user_query": "q", "messages": []}, None])
    async def test_unknown_or_empty_thread_is_a_fresh_run(self, values):
        plan = await plan_run(self._graph(SimpleNamespace(next=(), interrupts=(), values=values)), "hi", "user_x")
        assert plan.graph_input["history"] == []
        assert plan.graph_input["previous_trip"] == {}

    @pytest.mark.asyncio
    async def test_no_thread_id_never_inspects_state(self):
        graph = self._graph(None)
        plan = await plan_run(graph, "hi", None)
        graph.aget_state.assert_not_awaited()
        assert plan.graph_input["history"] == []

    @pytest.mark.asyncio
    async def test_state_lookup_failure_falls_back_to_fresh_run(self):
        graph = SimpleNamespace(aget_state=AsyncMock(side_effect=RuntimeError("db down")))
        plan = await plan_run(graph, "hi", "user_x")
        assert plan.resumed is False
        assert plan.graph_input["history"] == []


class TestConversationEndToEnd:
    @pytest.mark.asyncio
    async def test_follow_up_carries_context_through_the_graph(self, graph_env):
        graph_env["extract"].return_value = ("full_trip", TOKYO)
        graph_env["follow"].return_value = ("full_trip", TripSlots(budget="budget"), True)

        first = await navigo_agent.run_travel_agent("Plan a 5-day trip to Tokyo from Dhaka")
        assert first["follow_up"] is False
        assert first["answer"] == "FINAL PLAN"

        second = await navigo_agent.run_travel_agent("make it cheaper", thread_id=first["thread_id"])
        assert second["follow_up"] is True
        assert second["needs_clarification"] is False     # destination/duration carried over
        assert second["thread_id"] != first["thread_id"]
        assert second["answer"] == "FINAL PLAN #2"

        # The follow-up extractor saw the first turn and its slots
        query, history, previous_trip = graph_env["follow"].await_args.args
        assert query == "make it cheaper"
        assert history == [{"user": "Plan a 5-day trip to Tokyo from Dhaka", "assistant": "FINAL PLAN"}]
        assert previous_trip["destination"] == "Tokyo"

        # Agents, supervisor and the final synthesizer all got the conversation
        flight_first, flight_second = graph_env["agent_messages"]["flight_agent"]
        assert "Conversation So Far" not in flight_first
        assert "Conversation So Far" in flight_second
        assert "Plan a 5-day trip to Tokyo from Dhaka" in flight_second
        assert "Conversation So Far" in graph_env["supervisor_prompts"][-1]
        final_prompt = graph_env["final_prompts"][-1]
        assert "This is a follow-up" in final_prompt
        assert "- Budget: budget" in final_prompt
        assert "- Destination: Tokyo" in final_prompt
        assert "This is a follow-up" not in graph_env["final_prompts"][0]

    @pytest.mark.asyncio
    async def test_history_accumulates_across_turns(self, graph_env):
        graph_env["extract"].return_value = ("weather", TripSlots(destination="Tokyo"))
        graph_env["follow"].return_value = ("weather", TripSlots(), True)

        result = await navigo_agent.run_travel_agent("Weather in Tokyo?")
        for question in ("And next week?", "What should I pack?"):
            result = await navigo_agent.run_travel_agent(question, thread_id=result["thread_id"])

        history = graph_env["follow"].await_args.args[1]
        assert [turn["user"] for turn in history] == ["Weather in Tokyo?", "And next week?"]

    @pytest.mark.asyncio
    async def test_new_topic_is_independent(self, graph_env):
        graph_env["extract"].return_value = ("full_trip", TOKYO)
        graph_env["follow"].return_value = ("weather", TripSlots(destination="Paris"), False)

        first = await navigo_agent.run_travel_agent("Plan a 5-day trip to Tokyo from Dhaka")
        second = await navigo_agent.run_travel_agent("Weather in Paris?", thread_id=first["thread_id"])

        assert second["follow_up"] is False
        graph_env["weather"].assert_awaited_with("Paris")
        assert "Conversation So Far" not in graph_env["final_prompts"][-1]
        assert "Dhaka" not in graph_env["final_prompts"][-1]

    @pytest.mark.asyncio
    async def test_clarification_reply_is_not_treated_as_follow_up(self, graph_env):
        """A paused thread is resumed — the follow-up path must not kick in."""
        graph_env["extract"].side_effect = [
            ("weather", TripSlots()),
            ("weather", TripSlots(destination="Tokyo")),
        ]
        first = await navigo_agent.run_travel_agent("What's the weather like?")
        assert first["needs_clarification"] is True
        second = await navigo_agent.run_travel_agent("Tokyo", thread_id=first["thread_id"])
        assert second["thread_id"] == first["thread_id"]
        assert second["follow_up"] is False
        graph_env["follow"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_follow_up_after_clarified_turn_keeps_merged_query(self, graph_env):
        graph_env["extract"].side_effect = [
            ("weather", TripSlots()),
            ("weather", TripSlots(destination="Tokyo")),
        ]
        graph_env["follow"].return_value = ("weather", TripSlots(), True)
        first = await navigo_agent.run_travel_agent("What's the weather like?")
        second = await navigo_agent.run_travel_agent("Tokyo", thread_id=first["thread_id"])
        await navigo_agent.run_travel_agent("and tomorrow?", thread_id=second["thread_id"])
        history = graph_env["follow"].await_args.args[1]
        assert "Additional details from the user: Tokyo" in history[0]["user"]

    @pytest.mark.asyncio
    async def test_streaming_follow_up(self, graph_env):
        graph_env["extract"].return_value = ("weather", TripSlots(destination="Tokyo"))
        graph_env["follow"].return_value = ("weather", TripSlots(), True)

        def complete(chunks):
            last = chunks[-1].strip().split("\n")
            assert last[0] == "event: complete"
            return json.loads(last[1].removeprefix("data: "))

        first = complete([c async for c in streaming.stream_travel_plan("Weather in Tokyo?")])
        assert first["follow_up"] is False
        second = complete([c async for c in streaming.stream_travel_plan("And tomorrow?", first["thread_id"])])
        assert second["follow_up"] is True
        assert second["thread_id"] != first["thread_id"]
