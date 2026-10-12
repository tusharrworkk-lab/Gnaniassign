"""Regression tests for routing, memory, and failed-turn recovery."""

from __future__ import annotations

import unittest

from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.memory import ChatTurn, render_memory, select_memory, topic_from_query
from gnaniassignment.llm.prompts.intents import INTENT_KEYS
from gnaniassignment.llm.retrieval import IntentScopedRetriever
from gnaniassignment.orchestration.graph import build_graph, select_intent
from gnaniassignment.orchestration.session import TurnSession


class Router:
    def __init__(self, chosen: str = "user_core", score: float = 0.9) -> None:
        self.chosen, self.score, self.fail = chosen, score, False
        self.calls = 0
        self.states: list[str] = []

    def predict(self, state, questions, model=None):
        self.calls += 1
        self.states.append(state)
        if self.fail:
            raise RuntimeError("temporary provider error")
        return {"answers": {name: {"noul": self.score if name == self.chosen else 0.01} for name in questions}}


class Chat:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return '{"answer":"We operate in India.","memory_summary":"Coverage: operating region is India."}'


class Store:
    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    def search(self, query: str, limit: int = 4) -> list[str]:
        self.calls += 1
        return [self.text]


class PipelineTests(unittest.TestCase):
    def test_low_confidence_clarifies_without_chat_call(self):
        router, chat = Router(score=0.2), Chat()
        session = TurnSession(build_graph(router, enquiry_agents(chat), threshold=0.5))
        result = session.ask("Tell me about it")
        self.assertEqual(result["intent"], "clarify")
        self.assertIn("clarify", result["spoken"].lower())
        self.assertEqual(chat.prompts, [])

    def test_retry_keeps_buffered_query_and_does_not_duplicate_history(self):
        router, chat = Router(), Chat()
        session = TurnSession(build_graph(router, enquiry_agents(chat)))
        session.feed_final("Where do you operate?")
        router.fail = True
        with self.assertRaises(RuntimeError):
            session.feed_end()
        self.assertEqual(session.phrases, ["Where do you operate?"])
        self.assertEqual(session.history, [])
        router.fail = False
        result = session.feed_end()
        self.assertEqual(result["spoken"], "We operate in India.")
        self.assertEqual(len(session.history), 1)

    def test_memory_is_relevant_bounded_and_chronological(self):
        history = [
            ChatTurn("Where?", "India", "Coverage: India", "user_core", "coverage"),
            ChatTurn("Why?", "Because", "Reason", "product_why", "value"),
            ChatTurn("Regions?", "India", "Regions: India", "user_core", "coverage"),
            ChatTurn("Price?", "Unknown", "Price unknown", "product_what", "product"),
            ChatTurn("How?", "Online", "Process online", "user_operations", "operations"),
        ]
        selected = select_memory(history, "user_core", "coverage")
        self.assertEqual([turn.query for turn in selected], ["Where?", "Regions?", "Price?", "How?"])
        self.assertEqual(render_memory(selected).count("Memory:"), 4)
        self.assertEqual(topic_from_query("Somewhere else?"), "general")

    def test_invalid_jev_scores_are_rejected(self):
        scores = {name: {"noul": 0.1} for name in INTENT_KEYS}
        scores["user_core"]["noul"] = float("nan")
        with self.assertRaises(ValueError):
            select_intent(scores)

    def test_classifier_receives_only_two_recent_summaries(self):
        router, chat = Router(), Chat()
        session = TurnSession(build_graph(router, enquiry_agents(chat)))
        session.ask("Where do you operate?")
        session.ask("Which regions?")
        session.ask("And there?")
        self.assertEqual(router.states[-1].count("Memory:"), 2)
        self.assertIn("Current user query: And there?", router.states[-1])

    def test_only_selected_intent_store_is_queried(self):
        core, product = Store("Company fact"), Store("Product fact")
        chat = Chat()
        retriever = IntentScopedRetriever({"user_core": core, "product_what": product})
        session = TurnSession(build_graph(Router(), enquiry_agents(chat, retriever=retriever)))
        session.ask("Where do you operate?")
        self.assertEqual((core.calls, product.calls), (1, 0))
        self.assertIn("Company fact", chat.prompts[0])
        self.assertNotIn("Product fact", chat.prompts[0])


if __name__ == "__main__":
    unittest.main()
