"""Regression tests for routing, memory, and failed-turn recovery."""

from __future__ import annotations

import asyncio
import io
import json
import unittest
from types import SimpleNamespace

from gnaniassignment.llm.agents.enquiry import enquiry_agents
from gnaniassignment.llm.chat import BedrockChatModel, BedrockGptOssChatModel, CerebrasChatModel, DEFAULT_BEDROCK_MODEL, DEFAULT_BEDROCK_GPT_OSS_MODEL, DEFAULT_CEREBRAS_MODEL, DEFAULT_GROQ_MODEL, DEFAULT_SIMPLISMART_MODEL, DEFAULT_SUBAGENT_MODEL, GroqChatModel, SimplismartChatModel, VercelGatewayChatModel
from gnaniassignment.llm.memory import ChatTurn, render_memory, select_memory, topic_from_query
from gnaniassignment.llm.local_laya import LocalLayaRouter, require_requested_cuda
from gnaniassignment.llm.laya_cpp import DEFAULT_LAYA_CPP_MODEL, DEFAULT_LAYA_CPP_URL, LayaCppRouter
from gnaniassignment.llm.laya_lambda import LayaLambdaRouter
from gnaniassignment.llm.agents.retail import retail_clarification_agent
from gnaniassignment.llm.prompts.intents import QUESTIONS
from gnaniassignment.llm.retrieval import IntentScopedRetriever
from gnaniassignment.llm.router import LAYA_GATEWAY_URL, LAYA_MODEL, LayaGatewayRouter
from gnaniassignment.app.config import Settings
from gnaniassignment.llm.agents.generic import generic_agents
from gnaniassignment.llm.agents.generic import GENERIC_PROMPT
from gnaniassignment.llm.generic_router import GenericRouter
from gnaniassignment.conversation_analytics import BedrockConversationJudge, flow_flags
from gnaniassignment.orchestration.graph import build_graph, select_intent
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.pipeline.provider_benchmark import provider_from_event
from gnaniassignment.pipeline.gnani_microphone import pcm_frames
from gnaniassignment.pipeline.laya_benchmark import _same_execution_device
from gnaniassignment.stt.gnani import GnaniProcessing, GnaniRealtimeStt, GnaniTranscript, GnaniTurnBridge, PCM_FRAME_BYTES
from gnaniassignment.tts.gnani import GnaniRealtimeTts, decorate_tts_text, should_add_clear_throat
from gnaniassignment.voice.providers import STT_PROVIDER_FACTORIES, TTS_PROVIDER_FACTORIES, create_stt_provider, create_tts_provider
from gnaniassignment.prompts.tree import PromptNode, PromptPlan, PromptPlanResolver


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
    def test_voicelink_reply_metrics_separate_tts_from_websocket_handoff(self):
        from gnaniassignment.telephony.voicelink import VoiceLinkCall
        from gnaniassignment.tts.gnani import GnaniAudioChunk

        class WebSocket:
            def __init__(self):
                self.messages = []

            async def send_json(self, message):
                self.messages.append(message)

        class Tts:
            sample_rate = 8000
            last_first_audio_ms = None
            last_synthesis_ms = None

            async def synthesize(self, text, **_kwargs):
                self.last_first_audio_ms = 7.0
                yield GnaniAudioChunk(b"\\x00" * 320, 0, True)
                self.last_synthesis_ms = 9.0

        async def run():
            socket, metrics = WebSocket(), []
            call = VoiceLinkCall(socket, "stream-1", "call-1", "caller", "customer", object(), Tts(), object())

            async def record(**values):
                metrics.append(values)

            call.metric = record
            await call.speak_reply("Hello")
            return socket.messages, metrics

        messages, metrics = asyncio.run(run())
        reply = metrics[-1]["latest_reply"]
        self.assertEqual(messages[-1], {"event": "mark", "stream_sid": "stream-1", "mark": {"name": "response_done"}})
        self.assertGreater(reply["outbound_frames_sent"], 0)
        self.assertIn("tts_first_audio_ms", reply)
        self.assertIn("outbound_first_frame_sent_ms", reply)
        self.assertIn("outbound_mark_sent_ms", reply)
        media = next(message for message in messages if message["event"] == "media")
        self.assertEqual(media["stream_sid"], "stream-1")

    def test_voicelink_retries_tts_when_provider_completes_with_no_audio(self):
        from gnaniassignment.telephony.voicelink import VoiceLinkCall
        from gnaniassignment.tts.gnani import GnaniAudioChunk

        class WebSocket:
            def __init__(self): self.messages = []
            async def send_json(self, message): self.messages.append(message)

        class Tts:
            sample_rate = 8000
            last_first_audio_ms = None
            last_synthesis_ms = None

            def __init__(self): self.calls = self.reopens = 0
            async def synthesize(self, text, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.last_synthesis_ms = 1.0
                    return
                self.last_first_audio_ms = 2.0
                yield GnaniAudioChunk(b"\x00" * 320, 0, True)
                self.last_synthesis_ms = 4.0
            async def __aexit__(self, *args): self.reopens += 1
            async def __aenter__(self): self.reopens += 1; return self

        async def run():
            socket, tts = WebSocket(), Tts()
            call = VoiceLinkCall(socket, "stream-1", "call-1", "caller", "customer", object(), tts, object())
            await call.speak_reply("Hello")
            return socket.messages, tts

        messages, tts = asyncio.run(run())
        self.assertEqual(tts.calls, 2)
        self.assertEqual(tts.reopens, 2)
        self.assertTrue(any(message["event"] == "media" for message in messages))

    def test_voicelink_passes_clear_throat_tag_to_live_tts_once_requested(self):
        from gnaniassignment.telephony.voicelink import VoiceLinkCall
        from gnaniassignment.tts.gnani import GnaniAudioChunk

        class WebSocket:
            async def send_json(self, _message): pass

        class Tts:
            sample_rate = 8000
            last_first_audio_ms = 1.0
            last_synthesis_ms = 2.0

            def __init__(self): self.clear_throat = None
            async def synthesize(self, _text, *, add_clear_throat=False):
                self.clear_throat = add_clear_throat
                yield GnaniAudioChunk(b"\x00" * 320, 0, True)

        async def run():
            tts = Tts()
            call = VoiceLinkCall(WebSocket(), "stream-1", "call-1", "caller", "customer", object(), tts, object())
            call.metric = lambda **_values: asyncio.sleep(0)
            await call.speak_reply("Hello", add_clear_throat=True)
            return tts.clear_throat

        self.assertTrue(asyncio.run(run()))

    def test_smart_turn_can_commit_a_nonfinal_stt_hypothesis_once(self):
        from gnaniassignment.telephony.voicelink import VoiceLinkCall

        async def run():
            queue = asyncio.Queue()
            call = VoiceLinkCall(object(), "stream-1", "call-1", "caller", "customer", object(), object(), object())
            call.turn_queue = queue
            call.latest_transcript = GnaniTranscript("kal meri party hai", segment_id="partial-1", is_final=False)

            metrics = []

            async def metric(**values):
                metrics.append(values)

            call.metric = metric
            first = await call.commit_latest_transcript("smart_turn")
            second = await call.commit_latest_transcript("smart_turn")
            return first, second, queue.get_nowait(), metrics

        first, second, transcript, metrics = asyncio.run(run())
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(transcript.text, "kal meri party hai")
        self.assertEqual(metrics[-1]["stt_turn_committed_by"], "smart_turn")

    def test_audio_enhancement_runs_before_voice_link_vad_and_stt(self):
        import audioop
        import base64
        from gnaniassignment.telephony.voicelink import VoiceLinkCall
        from gnaniassignment.voice.audio_enhancement import AudioEnhancementResult

        class Stt:
            sample_rate = 16000
            frame_bytes = 1024

            def __init__(self):
                self.frames = []

            async def send_pcm(self, frame):
                self.frames.append(frame)

        class Enhancer:
            mode = "enabled"
            backend = "webrtc_apm"

            def __init__(self):
                self.input = b""

            def process_pcm_s16le(self, pcm):
                self.input = pcm
                return AudioEnhancementResult(b"\x01\x00" * (len(pcm) // 2), -42.0, -55.0, 0.8, 0.2)

        async def run():
            stt, enhancer = Stt(), Enhancer()
            call = VoiceLinkCall(object(), "stream-1", "call-1", "caller", "customer", stt, object(), object())
            call.audio_enhancer = enhancer
            metrics = []

            async def metric(**values):
                metrics.append(values)

            call.metric = metric
            pcm_8k = b"\x00\x00" * 160
            payload = base64.b64encode(audioop.lin2alaw(pcm_8k, 2)).decode("ascii")
            await call.feed_media(payload)
            await call.feed_media(payload)
            return stt, enhancer, metrics

        stt, enhancer, metrics = asyncio.run(run())
        self.assertEqual(len(enhancer.input), 640)
        self.assertTrue(stt.frames)
        self.assertEqual(stt.frames[0][:2], b"\x01\x00")
        self.assertEqual(metrics[-1]["audio_enhancement_mode"], "enabled")

    def test_retail_minilm_switch_uses_configured_fallback(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {
            "INTENT_PROVIDER": "retail_minilm",
            "RETAIL_MINILM_ENABLED": "false",
        }, clear=True):
            self.assertEqual(Settings.from_environment().intent_provider, "generic")

    def test_generic_mode_sends_every_query_to_general_agent(self):
        chat = Chat()
        result = TurnSession(build_graph(GenericRouter(), generic_agents(chat))).ask("Mera order late hai")
        self.assertEqual(result["intent"], "generic")
        self.assertEqual(len(chat.prompts), 1)

    def test_personal_assistant_prompt_discloses_identity_and_blocks_impersonation(self):
        self.assertIn("opted in", GENERIC_PROMPT)
        self.assertIn("Never claim to be the owner or a human", GENERIC_PROMPT)
        self.assertIn("explicitly authorized", GENERIC_PROMPT)
        self.assertIn("FIRST-REPLY DISCLOSURE", GENERIC_PROMPT)

    def test_personal_assistant_prompt_requires_feminine_first_person_hinglish(self):
        self.assertIn("FEMALE PERSONA", GENERIC_PROMPT)
        self.assertIn("main karungi", GENERIC_PROMPT)
        self.assertIn("karunga", GENERIC_PROMPT)

    def test_session_records_a_complete_conversation_turn_for_review(self):
        class ConversationStore:
            def __init__(self):
                self.records = []

            def append(self, record):
                self.records.append(record)

            def attach_followup(self, session_id, next_user_text):
                return []

        store = ConversationStore()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(Chat())), customer_id="customer-1", conversation_store=store)
        session.ask("Mera order late hai, order ID ORD-1042 hai")
        self.assertEqual(len(store.records), 1)
        self.assertEqual(store.records[0].user_text, "Mera order late hai, order ID ORD-1042 hai")
        self.assertEqual(store.records[0].routed_intent, "generic")

    def test_flow_flags_find_repeated_field_requests_and_english(self):
        flags = flow_flags("Please share your order ID so I can help.", {"order_id": "ORD-1042"})
        self.assertIn("repeated_known_field_request", flags)
        self.assertIn("english_only_response", flags)

    def test_call_finalization_keeps_stt_turns_and_judgment_together(self):
        class CallStore:
            def __init__(self):
                self.records = []
                self.finalized = None

            def append(self, record):
                self.records.append(record)

            def attach_followup(self, session_id, next_user_text):
                return []

            def finalize_call(self, *args):
                self.finalized = args

        class Judge:
            def judge(self, stt_segments, transcript):
                return {"stt_intent_judgment": {"segments": [{"intent": "track_order"}]}, "conversation_patterns": {"dos": [], "donts": []}}

        store = CallStore()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(Chat())), conversation_store=store, call_judge=Judge())
        session.record_stt_segment("Mera order kahan hai", "segment-1", 120)
        session.ask("Mera order kahan hai")
        judgment = session.finalize_call()
        self.assertEqual(judgment["stt_intent_judgment"]["segments"][0]["intent"], "track_order")
        self.assertEqual(store.finalized[2][0]["text"], "Mera order kahan hai")
        self.assertEqual(store.finalized[3][0]["user"], "Mera order kahan hai")

    def test_session_preloads_and_refreshes_customer_order_window(self):
        calls = []

        def order_window(query, working_memory):
            calls.append((query, dict(working_memory)))
            return f"Verified account order window: {query or 'initial'}"

        chat = Chat()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)), account_context_provider=order_window)
        self.assertEqual(calls[0][0], "")
        session.ask("GRC-10003 ka refund kahan hai?")
        self.assertEqual(calls[-1][0], "GRC-10003 ka refund kahan hai?")
        self.assertIn("Verified account order window", chat.prompts[0])

    def test_verified_matching_order_is_explicitly_marked_as_already_provided(self):
        chat = Chat()
        agent = generic_agents(chat)["generic"]
        agent.reply(
            "GRC-10003 ka refund kahan hai?", {},
            "Verified matching account order window (use only these records as facts):\norder ID GRC-10003; refund status verification pending.",
        )
        self.assertIn("Do not ask for an order ID", chat.prompts[0])

    def test_bedrock_intent_judge_receives_only_stt_segments(self):
        class Client:
            def __init__(self):
                self.requests = []

            def converse(self, **request):
                self.requests.append(request)
                return {"output": {"message": {"content": [{"text": '{"segments": []}'}]}}}

        client = Client()
        judge = BedrockConversationJudge.__new__(BedrockConversationJudge)
        judge.client, judge.model, judge.max_tokens = client, "test-model", 100
        judge.judge([{"text": "Mera order kahan hai", "segment_id": "s1"}], [{"user": "Mera order kahan hai", "assistant": "Your order is shipped"}])
        first_payload = client.requests[0]["messages"][0]["content"][0]["text"]
        second_payload = client.requests[1]["messages"][0]["content"][0]["text"]
        self.assertIn("Mera order kahan hai", first_payload)
        self.assertNotIn("Your order is shipped", first_payload)
        self.assertIn("Your order is shipped", second_payload)

    def test_retail_minilm_switch_is_enabled_by_default(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"INTENT_PROVIDER": "retail_minilm", "RETAIL_MINILM_ENABLED": "true"}, clear=True):
            self.assertEqual(Settings.from_environment().intent_provider, "retail_minilm")

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

    def test_gnani_blank_final_transcript_does_not_dispatch(self):
        class Session:
            def feed_final(self, phrase):
                raise AssertionError("blank STT must not enter the session")

            def feed_end(self, on_chunk=None):
                raise AssertionError("blank STT must not enter the LLM")

        self.assertIsNone(GnaniTurnBridge(Session()).handle(GnaniTranscript("   ")))

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

    def test_gnani_timbre_v3_tts_decodes_audio_chunks(self):
        captured: dict[str, object] = {}

        class Socket:
            def __init__(self):
                self.sent: list[str] = []
                self.messages = [
                    '{"type":"start","message":"Streaming started"}',
                    '{"type":"audio","data":{"chunk_index":1,"audio":"YWJj","is_final":false}}',
                    '{"type":"complete","data":{"chunk_index":2,"audio":"","is_final":true}}',
                ]

            async def send(self, message):
                self.sent.append(message)

            async def recv(self):
                return self.messages.pop(0)

            async def close(self):
                return None

        async def connect(endpoint, headers):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            socket = Socket()
            captured["socket"] = socket
            return socket

        async def synthesize():
            async with GnaniRealtimeTts(api_key="test-key", connect=connect) as tts:
                return [chunk async for chunk in tts.synthesize("Mera order kahan hai?")]

        chunks = asyncio.run(synthesize())
        self.assertEqual(chunks[0].audio, b"abc")
        self.assertEqual(chunks[0].chunk_index, 1)
        self.assertEqual(captured["headers"], {"X-API-Key-ID": "test-key", "Content-Type": "application/json"})
        request = json.loads(captured["socket"].sent[0])
        self.assertEqual(request["model"], "timbre-v3.0")
        self.assertEqual(request["voice"], "Shivani")
        self.assertEqual(request["language"], "hi-en")
        self.assertEqual(request["audio_config"]["container"], "raw")

    def test_tts_reopens_an_idle_closed_provider_socket_before_a_reply(self):
        connections = []

        class Socket:
            def __init__(self, closed=False):
                self.closed = closed
                self.sent = []
                self.messages = ['{"type":"complete"}']

            async def send(self, message):
                self.sent.append(message)

            async def recv(self):
                return self.messages.pop(0)

            async def close(self):
                self.closed = True

        async def connect(endpoint, headers):
            socket = Socket(closed=not connections)
            connections.append(socket)
            return socket

        async def synthesize():
            async with GnaniRealtimeTts(api_key="test-key", connect=connect) as tts:
                return [chunk async for chunk in tts.synthesize("Hello")]

        self.assertEqual(asyncio.run(synthesize()), [])
        self.assertEqual(len(connections), 2)
        self.assertEqual(len(connections[1].sent), 1)

    def test_tts_audio_tags_are_allowlisted_and_v3_only(self):
        spoken = decorate_tts_text("Aapka order ID ORD-1042 hai. <shouting> Dhanyavaad.", enabled=True, model="timbre-v3.0", add_clear_throat=True)
        self.assertIn("<clears throat>", spoken)
        self.assertIn("<giggle>", spoken)
        self.assertNotIn("<shouting>", spoken)
        self.assertNotIn("<pauses>", spoken)
        self.assertNotIn("<slow>", spoken)
        self.assertEqual(decorate_tts_text("Hello <giggle>", enabled=True, model="timbre-v2.5"), "Hello")

    def test_clear_throat_can_be_a_standalone_proactive_cue(self):
        self.assertEqual(
            decorate_tts_text("", enabled=True, model="timbre-v3.0", add_clear_throat=True),
            "<clears throat>",
        )

    def test_clear_throat_is_a_deterministic_ttft_threshold(self):
        self.assertFalse(should_add_clear_throat(700))
        self.assertTrue(should_add_clear_throat(701))
        self.assertFalse(should_add_clear_throat(None))

    def test_voice_provider_registry_has_explicit_adapter_names(self):
        self.assertIn("gnani", STT_PROVIDER_FACTORIES)
        self.assertIn("gnani", TTS_PROVIDER_FACTORIES)
        with self.assertRaisesRegex(ValueError, "Unsupported STT_PROVIDER"):
            create_stt_provider("not-a-provider")
        with self.assertRaisesRegex(ValueError, "Unsupported TTS_PROVIDER"):
            create_tts_provider("not-a-provider")

    def test_prompt_tree_selects_only_deterministic_matching_nodes_and_caches_l1(self):
        plan = PromptPlan(
            tenant_id="tenant-a", agent_id="generic", version=7,
            nodes=(
                PromptNode("root", None, {}, "Base", 10),
                PromptNode("voice", "root", {"channel": "voice"}, "Voice", 20),
                PromptNode("refund", "root", {"intent": "refund"}, "Refund", 30),
            ),
        )
        self.assertEqual(plan.render({"channel": "voice"}), "Base\n\nVoice")

        class Store:
            def __init__(self): self.calls = 0
            def load_active(self, tenant_id, agent_id):
                self.calls += 1
                return plan

        store = Store()
        resolver = PromptPlanResolver(store, ttl_seconds=60)
        self.assertEqual(resolver.resolve("tenant-a", "generic"), plan)
        self.assertEqual(resolver.resolve("tenant-a", "generic"), plan)
        self.assertEqual(store.calls, 1)

    def test_latency_giggle_renders_a_tag_only_tts_request(self):
        self.assertEqual(
            decorate_tts_text("<giggle>", enabled=True, model="timbre-v3.0", latency_giggle=True),
            "<giggle>",
        )
        self.assertEqual(
            decorate_tts_text("<giggle>", enabled=False, model="timbre-v3.0", latency_giggle=True),
            "",
        )

    def test_gnani_16khz_raw_websocket_headers_and_events(self):
        captured: dict[str, object] = {}

        class Socket:
            def __init__(self):
                self.sent: list[bytes] = []
                self.closed = False
                self.messages = [
                    '{"type":"connected","config":{"sample_rate":16000,"chunk_size":512}}',
                    '{"type":"processing","timestamp":"2026-10-20T00:00:00Z"}',
                    '{"type":"transcript","text":"mera order kahan hai","segment_id":"s-1","segment_index":0,"audio_duration_ms":640,"latency":111}',
                ]

            async def send(self, frame):
                self.sent.append(frame)

            async def close(self):
                self.closed = True

            async def recv(self):
                return self.messages.pop(0)

            async def __aiter__(self):
                while self.messages:
                    yield self.messages.pop(0)

        socket = Socket()

        async def connect(endpoint, headers):
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            return socket

        async def exercise():
            stt = GnaniRealtimeStt(api_key="test-key", language_code="hi-IN", sample_rate=16000, min_silence_ms=100, connect=connect)
            self.assertEqual(stt.frame_interval_seconds, 0.032)
            async with stt:
                self.assertEqual(stt.active_sample_rate, 16000)
                await stt.send_pcm(bytes(PCM_FRAME_BYTES))
                return [event async for event in stt.events()]

        events = asyncio.run(exercise())
        headers = captured["headers"]
        self.assertEqual(captured["endpoint"], "wss://api.vachana.ai/stt/v3/stream")
        self.assertEqual(headers["x-sample-rate"], "16000")
        self.assertEqual(headers["x-min-silence-ms"], "100")
        self.assertEqual(socket.sent, [bytes(PCM_FRAME_BYTES)])
        self.assertTrue(socket.closed)
        self.assertEqual(events[0], GnaniProcessing(timestamp="2026-10-20T00:00:00Z"))
        self.assertEqual(events[1], GnaniTranscript("mera order kahan hai", "s-1", 0, 640, 111))

    def test_microphone_pcm_frames_coalesce_partial_alsa_reads(self):
        async def exercise():
            reader = asyncio.StreamReader()
            reader.feed_data(b"a" * 512)
            reader.feed_data(b"b" * 1536)
            reader.feed_eof()
            return [frame async for frame in pcm_frames(reader)]

        frames = asyncio.run(exercise())
        self.assertEqual([len(frame) for frame in frames], [1024, 1024])
        self.assertEqual(frames[0], b"a" * 512 + b"b" * 512)
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
        self.assertIn("order", result["spoken"].lower())
        self.assertEqual(chat.prompts, [])

    def test_low_confidence_uses_configured_clarification_subagent(self):
        router, chat = Router(confidence=0.2), Chat()
        session = TurnSession(
            build_graph(
                router,
                enquiry_agents(chat),
                threshold=0.5,
                clarification_agent=retail_clarification_agent(chat),
            )
        )
        result = session.ask("Kya haal hai bhai?")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(result["spoken"], "We operate in India.")
        self.assertEqual(len(chat.prompts), 1)
        self.assertIn("Kya haal hai bhai?", chat.prompts[0])

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

    def test_generic_memory_excludes_unrelated_small_talk(self):
        history = [
            ChatTurn("GRC-10003 ka refund kahan hai", "Pending", "Refund pending", "generic", "general"),
            ChatTurn("car", "Order ID?", "Asked for order ID", "generic", "general"),
            ChatTurn("kya haal aapke", "Order ID?", "Asked for order ID", "generic", "general"),
        ]
        selected = select_memory(history, "generic", "general", "GRC-10003 ka refund kahan hai")
        self.assertEqual([turn.query for turn in selected], ["GRC-10003 ka refund kahan hai"])

    def test_cross_call_memory_requires_an_explicit_continuity_cue(self):
        history = [
            ChatTurn(
                "old", "old", "Previous call: caller asked Tushar to pass on a party message.",
                "generic", "general",
            ),
        ]
        self.assertEqual(
            select_memory(history, "generic", "general", "Kal mujhe ek naya message dena hai", require_continuity=True),
            [],
        )
        self.assertEqual(
            select_memory(history, "generic", "general", "Pichhli baar maine kya bola tha?", require_continuity=True),
            history,
        )

    def test_generic_prompt_prevents_repeated_greetings_on_continuing_calls(self):
        self.assertIn("Greet at most once per call", GENERIC_PROMPT)
        self.assertIn("continuing call", GENERIC_PROMPT)

    def test_generic_prompt_keeps_clear_message_acknowledgements_natural(self):
        self.assertIn("without quoting or restating the message", GENERIC_PROMPT)
        self.assertIn("Never repeat the same message", GENERIC_PROMPT)

    def test_session_restores_and_persists_episodic_memory(self):
        class MemoryStore:
            def __init__(self):
                self.loaded_for: tuple[str, int] | None = None
                self.saved: list[tuple[str, str, ChatTurn]] = []

            def load_recent(self, customer_id: str, limit: int) -> list[ChatTurn]:
                self.loaded_for = (customer_id, limit)
                return [ChatTurn("Earlier coverage question", "India", "Coverage: India", "user_core", "coverage")]

            def append(self, customer_id: str, session_id: str, turn: ChatTurn) -> None:
                self.saved.append((customer_id, session_id, turn))

        store = MemoryStore()
        session = TurnSession(build_graph(Router(), enquiry_agents(Chat())), memory_store=store, customer_id="mock-customer-001")
        self.assertEqual(store.loaded_for, ("mock-customer-001", 100))
        self.assertEqual(len(session.history), 1)
        session.ask("Where do you operate?")
        self.assertEqual(len(store.saved), 1)
        self.assertEqual(store.saved[0][0], "mock-customer-001")
        self.assertEqual(store.saved[0][2].intent, "user_core")

    def test_new_call_uses_compact_prior_summaries_and_keeps_active_turns(self):
        class MemoryStore:
            def __init__(self):
                self.saved_call_summaries = []

            def load_call_summaries(self, customer_id, limit):
                self.loaded = (customer_id, limit)
                return [ChatTurn("old", "old", "Previous call: caller asked about a meeting.", "generic", "general")]

            def append(self, customer_id, session_id, turn):
                pass

            def append_call_summary(self, customer_id, session_id, summary, intent, topic):
                self.saved_call_summaries.append((customer_id, summary, intent, topic))

        chat, store = Chat(), MemoryStore()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)), memory_store=store, customer_id="caller-1")
        session.ask("Tushar ko kal party ke baare mein bata dena")
        self.assertNotIn("Previous call: caller asked about a meeting.", chat.prompts[-1])
        session.ask("haan, bata do")
        self.assertEqual(store.loaded, ("caller-1", 3))
        self.assertIn("Current call context:\nMemory: Coverage: operating region is India.", chat.prompts[-1])
        session.finalize_call()
        self.assertEqual(len(store.saved_call_summaries), 1)
        self.assertIn("Previous call:", store.saved_call_summaries[0][1])

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

        router = LayaCppRouter(base_url=DEFAULT_LAYA_CPP_URL, api_key="local-key")
        router.model = DEFAULT_LAYA_CPP_MODEL
        router.http = HTTP()
        state = {"current_query": "Where do you operate?", "recent_conversation": []}
        result = router.predict(state, QUESTIONS)
        url, kwargs = router.http.request
        self.assertEqual(url, f"{DEFAULT_LAYA_CPP_URL}/v1/systemone")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local-key")
        self.assertEqual(kwargs["json"], {"model": DEFAULT_LAYA_CPP_MODEL, "state": state, "questions": QUESTIONS})
        self.assertEqual(result["answers"]["intent"]["choice"], "user_core")

    def test_laya_lambda_adapter_invokes_private_function(self):
        captured: dict[str, object] = {}

        class Client:
            def invoke(self, **kwargs):
                captured.update(kwargs)
                return {"Payload": io.BytesIO(b'{"statusCode":200,"body":"{\\\"answers\\\":{\\\"intent\\\":{\\\"choice\\\":\\\"track_order\\\"}}}"}')}

        router = LayaLambdaRouter(function_name="test-laya", client=Client())
        result = router.predict("mera order kahan hai", {"intent": {"type": "choice"}})
        self.assertEqual(result["answers"]["intent"]["choice"], "track_order")
        self.assertEqual(captured["FunctionName"], "test-laya")
        self.assertEqual(captured["InvocationType"], "RequestResponse")

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

    def test_cerebras_llama_stream_uses_low_voice_token_cap(self):
        class ChatResource:
            def create(self, **kwargs):
                self.request = kwargs
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="नमस्ते"))])])

        resource = ChatResource()
        model = CerebrasChatModel(model=DEFAULT_CEREBRAS_MODEL, client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        self.assertEqual(list(model.stream_complete("System", "User")), ["नमस्ते"])
        self.assertEqual(resource.request["model"], "llama-3.3-70b")
        self.assertEqual(resource.request["max_tokens"], 32)
        self.assertEqual(resource.request["reasoning_effort"], "none")
        self.assertIs(resource.request["stream"], True)

    def test_groq_llama_stream_uses_expected_model(self):
        class ChatResource:
            def create(self, **kwargs):
                self.request = kwargs
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="नमस्ते"))])])

        resource = ChatResource()
        model = GroqChatModel(model=DEFAULT_GROQ_MODEL, client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        self.assertEqual(list(model.stream_complete("System", "User")), ["नमस्ते"])
        self.assertEqual(resource.request["model"], "llama-3.3-70b-versatile")
        self.assertEqual(resource.request["max_tokens"], 32)
        self.assertEqual(resource.request["reasoning_effort"], "low")
        self.assertIs(resource.request["stream"], True)

    def test_simplismart_llama_stream_uses_expected_model(self):
        class ChatResource:
            def create(self, **kwargs):
                self.request = kwargs
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="नमस्ते"))])])

        resource = ChatResource()
        model = SimplismartChatModel(model=DEFAULT_SIMPLISMART_MODEL, client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        self.assertEqual(list(model.stream_complete("System", "User")), ["नमस्ते"])
        self.assertEqual(resource.request["model"], "meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.assertEqual(resource.request["max_tokens"], 32)
        self.assertIs(resource.request["stream"], True)

    def test_bedrock_gemma_stream_uses_mumbai_model(self):
        class BedrockResource:
            def converse_stream(self, **kwargs):
                self.request = kwargs
                return {"stream": iter([
                    {"contentBlockDelta": {"delta": {"text": "नमस्ते"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                ])}

        resource = BedrockResource()
        model = BedrockChatModel(model=DEFAULT_BEDROCK_MODEL, client=resource)
        self.assertEqual(list(model.stream_complete("System", "User")), ["नमस्ते"])
        self.assertEqual(resource.request["modelId"], "google.gemma-3-4b-it")
        self.assertEqual(resource.request["inferenceConfig"]["maxTokens"], 32)

    def test_bedrock_gpt_oss_stream_uses_dedicated_reasoning_configuration(self):
        class BedrockResource:
            def converse_stream(self, **kwargs):
                self.request = kwargs
                return {"stream": iter([
                    {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "internal"}}}},
                    {"contentBlockDelta": {"delta": {"text": "Final answer"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                ])}

        resource = BedrockResource()
        model = BedrockGptOssChatModel(model=DEFAULT_BEDROCK_GPT_OSS_MODEL, client=resource)
        self.assertEqual(list(model.stream_complete("System", "User")), ["Final answer"])
        self.assertEqual(resource.request["modelId"], "openai.gpt-oss-120b-1:0")
        self.assertEqual(resource.request["inferenceConfig"]["maxTokens"], 256)
        self.assertEqual(resource.request["additionalModelRequestFields"]["reasoning_effort"], "low")

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

    def test_memory_recall_acknowledgement_is_spoken_before_recalled_answer(self):
        prior = ChatTurn("old", "old", "Previous call: party at 6 PM.", "generic", "general")

        class MemoryStore:
            def load_call_summaries(self, customer_id, limit):
                return [prior]
            def append(self, customer_id, session_id, turn):
                pass

        class ToolChat:
            def stream_complete_with_memory_recall(self, system, user, recall_tool, on_recall_started):
                on_recall_started()
                self.recalled = recall_tool()
                yield '{"answer":"Pichhli baar aapne party ka zikr kiya tha.","memory_summary":"Recalled party."}'

            def stream_complete(self, system, user):
                yield '{"answer":"Fresh answer.","memory_summary":"Fresh."}'

        chunks: list[str] = []
        session = TurnSession(
            build_graph(GenericRouter(), generic_agents(ToolChat())),
            memory_store=MemoryStore(),
            customer_id="caller-1",
        )
        result = session.ask_stream("Pichhli baar kya baat hui thi?", chunks.append)
        self.assertTrue(chunks[0].startswith("Haan, ek pal"))
        self.assertEqual(result["spoken"], "Pichhli baar aapne party ka zikr kiya tha.")
        self.assertNotIn(chunks[0], result["spoken"])

    def test_clarification_is_delivered_to_stream_callback(self):
        session = TurnSession(build_graph(Router(chosen="other"), enquiry_agents(Chat())))
        chunks: list[str] = []
        result = session.ask_stream("Unclear", chunks.append)
        self.assertEqual(chunks, [result["spoken"]])


if __name__ == "__main__":
    unittest.main()
