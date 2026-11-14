"""Regression coverage for one-pass LLM dialogue state and answer streaming."""

from __future__ import annotations

import json
import unittest

from gnaniassignment.llm.agents.generic import GENERIC_PROMPT, generic_agents
from gnaniassignment.llm.generic_router import GenericRouter
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.session import TurnSession


def response(
    answer: str,
    summary: str = "Caller request is understood.",
    dialogue_state: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "answer": answer,
        "dialogue_state": dialogue_state or {
            "act": "request",
            "resolution": "answered",
            "clarification": "not_needed",
            "next_expected_act": "other",
        },
        "memory_summary": summary,
        "memory_operations": [],
    }


class ScriptedChat:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, _system: str, user: str) -> str:
        self.prompts.append(user)
        return json.dumps(self.responses[len(self.prompts) - 1])


class DialogueStateTests(unittest.TestCase):
    def make_session(self, chat: ScriptedChat) -> TurnSession:
        return TurnSession(build_graph(GenericRouter(), generic_agents(chat)))

    def test_every_turn_uses_the_existing_single_llm_call(self):
        chat = ScriptedChat([
            response("Theek hai."),
            response("Bilkul, take care.", dialogue_state={
                "act": "closing", "resolution": "closed",
                "clarification": "not_needed", "next_expected_act": "none",
            }),
        ])
        session = self.make_session(chat)

        session.ask("Accha")
        session.ask("Koi baat nahi")

        self.assertEqual(len(chat.prompts), 2)
        self.assertNotIn("deterministic", chat.prompts[0].lower())

    def test_delivered_llm_state_is_included_as_next_turn_context(self):
        first_state = {
            "act": "explicit_message_request",
            "resolution": "message_captured",
            "clarification": "not_needed",
            "next_expected_act": "correction_or_closing",
        }
        chat = ScriptedChat([
            response("Haan, samajh gayi. Main ye baat Tushar ke liye rakh lungi.", dialogue_state=first_state),
            response("Theek hai."),
        ])
        session = self.make_session(chat)

        session.ask("Tushar ko bata dena kal chhah baje party hai")
        session.ask("Accha")

        self.assertEqual(session.interaction_state["act"], "request")
        self.assertIn("Previous audible interaction state", chat.prompts[1])
        self.assertIn("previous caller act: explicit_message_request", chat.prompts[1])
        self.assertIn("likely next caller move: correction_or_closing", chat.prompts[1])

    def test_current_words_remain_primary_over_the_prior_hint(self):
        chat = ScriptedChat([
            response("Haan, samajh gayi.", dialogue_state={
                "act": "explicit_message_request", "resolution": "message_captured",
                "clarification": "not_needed", "next_expected_act": "correction_or_closing",
            }),
            response("Achha, theek hai. Main saat baje ko sahi detail maanungi.", dialogue_state={
                "act": "correction", "resolution": "clarification_resolved",
                "clarification": "resolved", "next_expected_act": "correction_or_closing",
            }),
        ])
        session = self.make_session(chat)
        session.ask("Tushar ko kal chhah baje party ke baare mein batana")

        session.ask("Nahi, saat baje hai")

        self.assertIn("current caller words take priority", chat.prompts[1])
        self.assertEqual(session.interaction_state["act"], "correction")

    def test_clarification_budget_is_context_not_a_second_model_request(self):
        chat = ScriptedChat([
            response("Aap kis work item ka update pooch rahi hain?", dialogue_state={
                "act": "request", "resolution": "clarification_requested",
                "clarification": "asked", "next_expected_act": "caller_response",
            }),
            response("Main pehle wale sawal ko repeat nahi karungi. Aap work item ID bol dijiye.", dialogue_state={
                "act": "acknowledgement", "resolution": "limitation",
                "clarification": "not_needed", "next_expected_act": "message_or_callback_detail",
            }),
        ])
        session = self.make_session(chat)
        session.ask("Update kya hai")
        session.ask("Accha")

        self.assertEqual(len(chat.prompts), 2)
        self.assertIn("clarification turns already used for that need: 1", chat.prompts[1])
        self.assertIn("do not ask that clarification again", chat.prompts[1])

    def test_unheard_response_never_promotes_its_dialogue_state(self):
        chat = ScriptedChat([response("Aap kis kaam ke baare mein pooch rahi hain?", dialogue_state={
            "act": "request", "resolution": "clarification_requested",
            "clarification": "asked", "next_expected_act": "caller_response",
        })])
        session = self.make_session(chat)

        session.ask_stream("Status kya hai", lambda _chunk: None, defer_delivery=True)
        self.assertEqual(session.interaction_state, {})
        session.mark_delivery("failed")

        self.assertEqual(session.interaction_state, {})

    def test_prompt_requires_answer_first_and_llm_generated_dialogue_state(self):
        self.assertIn('"answer" first, "dialogue_state" second', GENERIC_PROMPT)
        self.assertIn("Infer the caller's act", GENERIC_PROMPT)
        self.assertIn("short complete clause and punctuation", GENERIC_PROMPT)
        self.assertNotIn("DIALOGUE POLICY", GENERIC_PROMPT)

    def test_answer_text_reaches_callback_before_later_json_fields_arrive(self):
        callbacks: list[str] = []

        class StreamingChat:
            def stream_complete(self, _system, _user):
                yield '{"answer":"Haan'
                self.assertEqual(callbacks, ["Haan"])
                yield ', samajh gayi.",'
                self.assertEqual("".join(callbacks), "Haan, samajh gayi.")
                yield ('"dialogue_state":{"act":"acknowledgement","resolution":"acknowledged",'
                       '"clarification":"not_needed","next_expected_act":"other"},'
                       '"memory_summary":"Caller acknowledged the answer.","memory_operations":[]}')

            def assertEqual(self, left, right):
                if left != right:
                    raise AssertionError(f"{left!r} != {right!r}")

        agent = generic_agents(StreamingChat())["generic"]
        result = agent.run_stream("Accha", {}, "", callbacks.append)

        self.assertEqual(result.answer, "Haan, samajh gayi.")
        self.assertEqual(result.dialogue_state["act"], "acknowledgement")


if __name__ == "__main__":
    unittest.main()
