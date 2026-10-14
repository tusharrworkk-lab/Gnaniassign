"""Regression tests for routing, memory, and failed-turn recovery."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import DEFAULT_SUBAGENT_MODEL, VercelGatewayChatModel
from gnaniassignment.llm.memory import ChatTurn, render_memory, select_memory, topic_from_query
from gnaniassignment.llm.local_laya import LocalLayaRouter, require_requested_cuda
from gnaniassignment.llm.laya_cpp import DEFAULT_LAYA_CPP_MODEL, DEFAULT_LAYA_CPP_URL, LayaCppRouter
from gnaniassignment.llm.prompts.intents import QUESTIONS
from gnaniassignment.llm.retrieval import IntentScopedRetriever
from gnaniassignment.llm.router import LAYA_GATEWAY_URL, LAYA_MODEL, LayaGatewayRouter
from gnaniassignment.orchestration.graph import build_graph, select_intent
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.pipeline.provider_benchmark import provider_from_event
from gnaniassignment.pipeline.laya_benchmark import _same_execution_device
from gnaniassignment.stt.gnani import GnaniTranscript, GnaniTurnBridge, PCM_FRAME_BYTES


class Router:
    def __init__(self, chosen: str = "user_core", confidence: float = 0.9) -> None:
        self.chosen, self.confidence, self.fail = chosen, confidence, False
        self.calls = 0
        self.states: list[dict] = []

    def predict(self, state, questions, model=None):
        self.calls += 1
        self.states.append(state)
        if self.fail:
            raise RuntimeError("temporary provider error")
        return {"answers": {"intent": {"type": "choice", "choice": self.chosen, "confidence": self.confidence}}}


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
    def test_gnani_final_transcript_auto_dispatches_once_by_default(self):
        class Session:
            def __init__(self):
                self.phrases: list[str] = []
                self.ends = 0

            def feed_final(self, phrase):
                self.phrases.append(phrase)
                return {"pending_text": " ".join(self.phrases)}

            def feed_end(self, on_chunk=None):
                self.ends += 1
                return {"spoken": "Done"}

        session = Session()
        bridge = GnaniTurnBridge(session)
        event = GnaniTranscript("Where do you operate?", segment_id="segment-1")
        self.assertEqual(bridge.handle(event), {"spoken": "Done"})
        self.assertIsNone(bridge.handle(event))
        self.assertEqual(session.ends, 1)

    def test_gnani_bridge_can_join_final_segments_when_requested(self):
        class Session:
            def __init__(self):
                self.phrases: list[str] = []

            def feed_final(self, phrase):
                self.phrases.append(phrase)
                return {"pending_text": " ".join(self.phrases)}

            def feed_end(self, on_chunk=None):
                return {"spoken": " ".join(self.phrases)}

        bridge = GnaniTurnBridge(Session(), auto_end_turn=False)
        self.assertEqual(bridge.handle(GnaniTranscript("Book a meeting.")), {"pending_text": "Book a meeting."})

    def test_gnani_pcm_frame_size_is_fixed(self):
        self.assertEqual(PCM_FRAME_BYTES, 1024)
    def test_laya_benchmark_treats_cuda_and_cuda_zero_as_same_device(self):
        self.assertTrue(_same_execution_device("cuda", "cuda:0"))
    def test_requested_cuda_fails_fast_when_runtime_is_unavailable(self):
        import sys

        original = sys.modules.get("torch")
        unavailable_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda="13.0"),
        )
        sys.modules["torch"] = unavailable_torch
        try:
            with self.assertRaisesRegex(RuntimeError, "cannot access an NVIDIA GPU"):
                require_requested_cuda("cuda")
        finally:
            if original is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = original

    def test_provider_benchmark_reads_gateway_metadata(self):
        event = {"provider_metadata": {"gateway": {"routing": {"finalProvider": "bedrock"}}}}
        self.assertEqual(provider_from_event(event), "bedrock")
        nested = {"choices": [{"delta": {"provider_metadata": {"gateway": {"routing": {"finalProvider": "novita"}}}}}]}
        self.assertEqual(provider_from_event(nested), "novita")

    def test_low_confidence_clarifies_without_chat_call(self):
        router, chat = Router(confidence=0.2), Chat()
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

    def test_invalid_laya_confidence_is_rejected(self):
        scores = {"intent": {"choice": "user_core", "confidence": float("nan")}}
        with self.assertRaises(ValueError):
            select_intent(scores)

    def test_classifier_receives_only_two_recent_summaries(self):
        router, chat = Router(), Chat()
        session = TurnSession(build_graph(router, enquiry_agents(chat)))
        session.ask("Where do you operate?")
        session.ask("Which regions?")
        session.ask("And there?")
        self.assertEqual(len(router.states[-1]["recent_conversation"]), 2)
        self.assertEqual(router.states[-1]["current_query"], "And there?")

    def test_other_choice_clarifies(self):
        chat = Chat()
        session = TurnSession(build_graph(Router(chosen="other"), enquiry_agents(chat)))
        self.assertEqual(session.ask("Unrelated question")["intent"], "clarify")
        self.assertEqual(chat.prompts, [])

    def test_only_selected_intent_store_is_queried(self):
        core, product = Store("Company fact"), Store("Product fact")
        chat = Chat()
        retriever = IntentScopedRetriever({"user_core": core, "product_what": product})
        session = TurnSession(build_graph(Router(), enquiry_agents(chat, retriever=retriever)))
        session.ask("Where do you operate?")
        self.assertEqual((core.calls, product.calls), (1, 0))
        self.assertIn("Company fact", chat.prompts[0])
        self.assertNotIn("Product fact", chat.prompts[0])

    def test_laya_adapter_sends_choice_question_to_vercel(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"answers": {"intent": {"type": "choice", "choice": "user_core", "confidence": 0.9}}}

        class HTTP:
            def __init__(self):
                self.request = None

            def post(self, url, **kwargs):
                self.request = (url, kwargs)
                return Response()

        router = LayaGatewayRouter(api_key="test-key")
        router.http = HTTP()
        state = {"current_query": "Where do you operate?", "recent_conversation": []}
        result = router.predict(state, QUESTIONS)
        url, kwargs = router.http.request
        self.assertEqual(url, LAYA_GATEWAY_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"], {"model": LAYA_MODEL, "state": state, "questions": QUESTIONS})
        self.assertEqual(result["answers"]["intent"]["choice"], "user_core")

    def test_local_laya_adapter_uses_loaded_agent(self):
        class Agent:
            def predict(self, state, questions):
                self.request = (state, questions)
                return {"answers": {"intent": {"choice": "product_what", "confidence": 0.8}}}

        agent = Agent()
        router = LocalLayaRouter(agent=agent, model_id="convaiinnovations/laya", device="cuda")
        state = {"current_query": "What does it do?", "recent_conversation": []}
        result = router.predict(state, QUESTIONS)
        self.assertEqual(agent.request, (state, QUESTIONS))
        self.assertEqual(result["answers"]["intent"]["choice"], "product_what")
        self.assertGreaterEqual(router.last_ms, 0)
        with self.assertRaises(ValueError):
            router.predict(state, QUESTIONS, model="different-checkpoint")

    def test_laya_cpp_adapter_sends_system_one_request_to_local_server(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"model": "laya", "answers": {"intent": {"choice": "user_core", "confidence": 0.9}}}

        class HTTP:
            def __init__(self):
                self.request = None

            def post(self, url, **kwargs):
                self.request = (url, kwargs)
                return Response()

        router = LayaCppRouter(api_key="local-key")
        router.http = HTTP()
        state = {"current_query": "Where do you operate?", "recent_conversation": []}
        result = router.predict(state, QUESTIONS)
        url, kwargs = router.http.request
        self.assertEqual(url, f"{DEFAULT_LAYA_CPP_URL}/v1/systemone")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local-key")
        self.assertEqual(kwargs["json"], {"model": DEFAULT_LAYA_CPP_MODEL, "state": state, "questions": QUESTIONS})
        self.assertEqual(result["answers"]["intent"]["choice"], "user_core")

    def test_llama_subagent_uses_vercel_gateway_client(self):
        class ChatResource:
            def __init__(self):
                self.request = None

            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer  "))])

        resource = ChatResource()
        client = SimpleNamespace(chat=SimpleNamespace(completions=resource))
        model = VercelGatewayChatModel(model=DEFAULT_SUBAGENT_MODEL, client=client)
        self.assertEqual(model.complete("System instructions", "User query"), "Answer")
        self.assertEqual(resource.request["model"], "meta/llama-3.1-8b")
        self.assertEqual(resource.request["messages"], [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "User query"},
        ])
        self.assertEqual(resource.request["max_tokens"], 220)

    def test_vercel_stream_yields_only_text_deltas(self):
        class ChatResource:
            def create(self, **kwargs):
                self.request = kwargs
                return iter([
                    SimpleNamespace(choices=[]),
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))]),
                ])

        resource = ChatResource()
        model = VercelGatewayChatModel(client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        self.assertEqual(list(model.stream_complete("System", "User")), ["Hello", " there"])
        self.assertIs(resource.request["stream"], True)

    def test_session_streams_decoded_answer_and_stores_summary(self):
        class StreamingChat:
            def stream_complete(self, system, user):
                for chunk in ['{"answer":"Hello \\', '"world\\".\\', 'nDone",', '"memory_summary":"Greeting saved."}']:
                    yield chunk

        session = TurnSession(build_graph(Router(), enquiry_agents(StreamingChat())))
        chunks: list[str] = []
        result = session.ask_stream("Greet me", chunks.append)
        self.assertEqual("".join(chunks), 'Hello "world".\nDone')
        self.assertEqual(result["spoken"], "".join(chunks))
        self.assertEqual(session.history[0].summary, "Greeting saved.")

    def test_clarification_is_delivered_to_stream_callback(self):
        session = TurnSession(build_graph(Router(chosen="other"), enquiry_agents(Chat())))
        chunks: list[str] = []
        result = session.ask_stream("Unclear", chunks.append)
        self.assertEqual(chunks, [result["spoken"]])


if __name__ == "__main__":
    unittest.main()
