"""Call context keeps verified facts and only handed-off agent speech."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gnaniassignment.app.container import create_calendar_presence_window
from gnaniassignment.llm.agents.generic import generic_agents
from gnaniassignment.llm.generic_router import GenericRouter
from gnaniassignment.llm.retrieval import PostgresKnowledgeStore, _knowledge_search_query
from gnaniassignment.orchestration.graph import build_graph
from gnaniassignment.orchestration.call_facts import apply_memory_operations, extract_deterministic_facts, render_call_facts
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.order_data import customer_order_window
from gnaniassignment.telephony.voicelink import _collect_turn_segments, _reply_to_transcript
from gnaniassignment.voice.events import VoiceTranscript


class Chat:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, _system: str, user: str) -> str:
        self.prompts.append(user)
        return "Theek hai, main note kar leti hoon."


class SemanticChat:
    def __init__(self, summaries: list[str]) -> None:
        self.summaries = summaries
        self.prompts: list[str] = []

    def complete(self, _system: str, user: str) -> str:
        self.prompts.append(user)
        summary = self.summaries[len(self.prompts) - 1]
        return json.dumps({"answer": "Theek hai, main note kar leti hoon.", "memory_summary": summary})


class CallContextDeliveryTests(unittest.TestCase):
    def test_deterministic_call_facts_preserve_exact_values_and_corrections(self):
        facts = extract_deterministic_facts(
            {}, "Mera naam Rahul hai, kal 6 baje party hai aur callback chahiye", 1,
        )
        self.assertEqual(facts["caller_name"]["value"], "Rahul")
        self.assertEqual(facts["event_date"]["value"], "kal")
        self.assertEqual(facts["event_time"]["value"], "6 baje")
        self.assertIs(facts["callback_requested"]["value"], True)

        corrected = extract_deterministic_facts(facts, "Nahi, 7 baje. Callback nahi chahiye", 2)
        self.assertEqual(corrected["event_time"]["value"], "7 baje")
        self.assertEqual(corrected["event_time"]["previous_value"], "6 baje")
        self.assertEqual(corrected["event_time"]["supersedes_turn"], 1)
        self.assertEqual(corrected["event_time"]["status"], "corrected")
        self.assertIs(corrected["callback_requested"]["value"], False)

    def test_model_memory_operations_require_allowed_fields_and_current_evidence(self):
        operations = [
            {"operation": "set", "field": "message", "value": "Party is tomorrow", "evidence": "kal party hai"},
            {"operation": "set", "field": "owner_notified", "value": True, "evidence": "kal party hai"},
            {"operation": "set", "field": "caller_name", "value": "Rohit", "evidence": "mera naam Rohit hai"},
        ]
        facts, audit = apply_memory_operations({}, operations, "kal party hai", 1)
        self.assertEqual(facts["message"]["value"], "Party is tomorrow")
        self.assertNotIn("owner_notified", facts)
        self.assertNotIn("caller_name", facts)
        self.assertEqual([entry["accepted"] for entry in audit], [True, False, False])
        self.assertEqual(audit[1]["reason"], "unsupported_field")
        self.assertEqual(audit[2]["reason"], "evidence_not_in_current_transcript")

    def test_session_combines_structured_facts_with_rolling_semantic_memo(self):
        class StructuredChat:
            def __init__(self): self.prompts = []
            def complete(self, _system, user):
                self.prompts.append(user)
                if len(self.prompts) == 1:
                    return json.dumps({
                        "answer": "Theek hai, main note kar leti hoon.",
                        "memory_summary": "Caller wants Tushar informed about tomorrow's 6 PM party.",
                        "memory_operations": [{
                            "operation": "set", "field": "message",
                            "value": "Party is tomorrow at 6 PM", "evidence": "kal 6 baje party hai",
                        }],
                    })
                return json.dumps({
                    "answer": "Theek hai, 7 baje note kar liya.",
                    "memory_summary": "Caller corrected tomorrow's party time to 7 PM.",
                    "memory_operations": [{
                        "operation": "replace", "field": "message",
                        "value": "Party is tomorrow at 7 PM", "evidence": "Nahi 7 baje",
                    }],
                })

        chat = StructuredChat()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)))
        session.ask("kal 6 baje party hai")
        self.assertEqual(session.call_facts["message"]["value"], "Party is tomorrow at 6 PM")
        session.ask("Nahi, 7 baje")

        self.assertIn("- event_time: 7 baje (corrected)", chat.prompts[-1])
        self.assertIn("- message: Party is tomorrow at 6 PM (confirmed)", chat.prompts[-1])
        self.assertEqual(session.call_facts["message"]["value"], "Party is tomorrow at 7 PM")
        self.assertEqual(session.call_facts["event_time"]["previous_value"], "6 baje")
        self.assertIn("Structured call facts", render_call_facts(session.call_facts))

    def test_provider_final_skips_endpoint_grace(self):
        async def check():
            turns = asyncio.Queue()
            endpoint = asyncio.Event()
            endpoint.set()
            await turns.put(VoiceTranscript("Kal party hai", is_final=True))
            return await asyncio.wait_for(_collect_turn_segments(
                turns, endpoint, endpoint_mode=True, coalesce_ms=1100, grace_ms=1000,
            ), timeout=0.5)

        self.assertEqual([item.text for item in asyncio.run(check())], ["Kal party hai"])

    def test_nonfinal_hypothesis_keeps_grace_for_provider_final(self):
        async def check():
            turns = asyncio.Queue()
            endpoint = asyncio.Event()
            endpoint.set()
            await turns.put(VoiceTranscript("Kal party", segment_id="segment-1", is_final=False))

            async def provider_final():
                await asyncio.sleep(0.02)
                await turns.put(VoiceTranscript("Kal party hai", segment_id="segment-1", is_final=True))

            producer = asyncio.create_task(provider_final())
            segments = await _collect_turn_segments(
                turns, endpoint, endpoint_mode=True, coalesce_ms=1100, grace_ms=250,
            )
            await producer
            return segments

        self.assertEqual([item.text for item in asyncio.run(check())], ["Kal party hai"])

    def test_voicelink_marks_audio_sent_only_after_speech_handoff(self):
        class Session:
            def __init__(self): self.statuses = []

            def ask_stream(self, _text, on_chunk, *, defer_delivery):
                assert defer_delivery
                on_chunk("Theek hai.")
                return {"intent": "generic", "spoken": "Theek hai."}

            def mark_delivery(self, status):
                self.statuses.append(status)
                return True

            def record_stt_segment(self, *_args): pass

        class Call:
            def __init__(self):
                self.started_at = time.perf_counter()
                self.reply_interrupted = asyncio.Event()
                self.session = Session()
                self.stream_sid = "test-stream"
                self.customer_id = "test-caller"
                self.turn_index = 1
                self.spoken = []

            async def begin_turn_evaluation(self): pass
            async def finish_turn_evaluation(self, _outcome): pass
            async def metric(self, **_values): pass
            async def trace(self, *_args, **_values): pass
            def note_llm_first_token(self): pass

            async def speak_reply(self, text, **_kwargs):
                self.spoken.append(text)
                self.session.statuses.append("tts_handed_off")
                return True

        call = Call()
        with patch.dict("os.environ", {"GNANI_TTS_POST_RESPONSE_CLEAR_THROAT_ENABLED": "false"}):
            asyncio.run(_reply_to_transcript(call, VoiceTranscript("Kal party hai"), None))
        self.assertEqual(call.spoken, ["Theek hai."])
        self.assertEqual(call.session.statuses, ["tts_handed_off", "audio_sent"])

    def test_voicelink_passes_completed_fragment_on_barge_in(self):
        class Session:
            def __init__(self): self.delivery = None

            def ask_stream(self, _text, on_chunk, *, defer_delivery):
                assert defer_delivery
                on_chunk("Haan.")
                on_chunk(" Aur main aage bataungi.")
                return {"intent": "generic", "spoken": "Haan. Aur main aage bataungi."}

            def mark_delivery(self, status, *, handed_off_text=""):
                self.delivery = (status, handed_off_text)
                return True

            def record_stt_segment(self, *_args): pass

        class Call:
            def __init__(self):
                self.started_at = time.perf_counter()
                self.reply_interrupted = asyncio.Event()
                self.session = Session()
                self.stream_sid = "test-stream"
                self.customer_id = "test-caller"
                self.turn_index = 1
                self.spoken = []

            async def begin_turn_evaluation(self): pass
            async def finish_turn_evaluation(self, _outcome): pass
            async def metric(self, **_values): pass
            async def trace(self, *_args, **_values): pass
            def note_llm_first_token(self): pass

            async def speak_reply(self, text, **_kwargs):
                self.spoken.append(text)
                if len(self.spoken) == 2:
                    self.reply_interrupted.set()
                    return False
                return True

        call = Call()
        with patch.dict("os.environ", {"GNANI_TTS_POST_RESPONSE_CLEAR_THROAT_ENABLED": "false"}):
            asyncio.run(_reply_to_transcript(call, VoiceTranscript("Kal party hai"), None))
        self.assertEqual(call.spoken, ["Haan.", "Aur main aage bataungi."])
        self.assertEqual(call.session.delivery, ("interrupted", "Haan."))

    def test_interrupted_answer_is_not_replayed_as_heard(self):
        chat = Chat()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)))
        session.ask_stream("Kal chhah baje party hai", lambda _text: None, defer_delivery=True)
        self.assertEqual(session.delivery_statuses, ["generated"])
        self.assertEqual(session.call_context_turns, [])

        self.assertTrue(session.mark_delivery("interrupted"))
        session.ask("Bata dogi?")
        self.assertEqual(session.delivery_statuses, ["interrupted", "delivered"])
        self.assertIn("Caller said: Kal chhah baje party hai", chat.prompts[-1])
        self.assertIn("do not assume the caller heard it", chat.prompts[-1])
        self.assertNotIn("Answer: Theek hai, main note kar leti hoon", chat.prompts[-1])

    def test_interrupted_answer_keeps_only_completed_audio_fragments(self):
        chat = Chat()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)))
        session.ask_stream("Kal chhah baje party hai", lambda _text: None, defer_delivery=True)
        self.assertTrue(session.mark_delivery("interrupted", handed_off_text="Theek hai,"))
        self.assertEqual(session.call_context_turns[0].answer, "Theek hai,")
        self.assertEqual(session.handed_off_answers, ["Theek hai,"])
        session.ask("Bata dogi?")
        self.assertIn("Theek hai,", chat.prompts[-1])
        self.assertIn("rest of the generated reply was not handed off", chat.prompts[-1])
        self.assertNotIn("Answer: Theek hai, main note kar leti hoon", chat.prompts[-1])

    def test_sent_audio_becomes_next_turn_context(self):
        chat = Chat()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)))
        session.ask_stream("Kal chhah baje party hai", lambda _text: None, defer_delivery=True)
        self.assertTrue(session.mark_delivery("audio_sent"))
        session.ask("Theek hai")
        self.assertIn("Answer: Theek hai, main note kar leti hoon", chat.prompts[-1])

    def test_call_memo_is_one_rolling_semantic_state(self):
        chat = SemanticChat([
            "Caller wants Tushar informed about a party; time is pending.",
            "Caller wants Tushar informed about tomorrow's 6 PM party; callback is not requested.",
            "Caller corrected the party time to 7 PM tomorrow; callback is not requested.",
            "Caller confirmed the final request is tomorrow's 7 PM party notice; nothing remains pending.",
        ])
        session = TurnSession(build_graph(GenericRouter(), generic_agents(chat)))
        session.ask("Tushar ko party ke baare mein batana")
        session.ask("Kal chhah baje")
        session.ask("Nahi, saat baje")
        session.ask("Haan, bas itna hi")

        self.assertEqual(session.call_memo, chat.summaries[-1])
        self.assertIn(chat.summaries[1], chat.prompts[2])
        self.assertNotIn(chat.summaries[0], chat.prompts[2])
        self.assertLessEqual(len(session.call_memo), 320)

    def test_final_call_summary_uses_latest_semantic_memo_only(self):
        class Memory:
            def __init__(self): self.saved = []
            def load_call_summaries(self, *_args): return []
            def append(self, *_args): pass
            def append_call_summary(self, _customer_id, _session_id, summary, *_args):
                self.saved.append(summary)

        memory = Memory()
        chat = SemanticChat([
            "Caller requests a callback; timing is pending.",
            "Caller requests a callback tomorrow after 4 PM; no other action is pending.",
        ])
        session = TurnSession(
            build_graph(GenericRouter(), generic_agents(chat)),
            memory_store=memory, customer_id="caller-1",
        )
        session.ask("Tushar se callback chahiye")
        session.ask("Kal chaar baje ke baad")
        session.finalize_call()

        self.assertEqual(memory.saved, ["Previous call: " + chat.summaries[-1]])

    def test_next_call_summary_excludes_unheard_answer(self):
        class Memory:
            def __init__(self): self.summaries = []; self.turns = []
            def load_call_summaries(self, _customer_id, _limit): return []
            def append(self, *_args): self.turns.append(_args[-1])
            def append_call_summary(self, _customer_id, _session_id, summary, *_args):
                self.summaries.append(summary)

        memory = Memory()
        session = TurnSession(
            build_graph(GenericRouter(), generic_agents(Chat())),
            memory_store=memory, customer_id="caller-1",
        )
        session.ask_stream("Kal chhah baje party hai", lambda _text: None, defer_delivery=True)
        self.assertEqual(memory.turns, [])
        session.mark_delivery("failed")
        self.assertEqual(len(memory.turns), 1)
        session.finalize_call()
        self.assertIn("Kal chhah baje party hai", memory.summaries[0])
        self.assertNotIn("Answer: Theek hai", memory.summaries[0])

    def test_judge_sees_only_handed_off_speech_and_audit_keeps_generation(self):
        class Store:
            def __init__(self): self.saved = None
            def attach_followup(self, *_args): return []
            def append(self, *_args): pass
            def finalize_call(self, *_args): self.saved = _args[3]

        class Judge:
            def __init__(self): self.seen = None
            def judge(self, _segments, transcript):
                self.seen = transcript
                return {}

        store, judge = Store(), Judge()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(Chat())), conversation_store=store, call_judge=judge)
        session.ask_stream("Kal party hai", lambda _text: None, defer_delivery=True)
        session.mark_delivery("interrupted")
        session.finalize_call()
        self.assertEqual(judge.seen[0]["assistant"], "")
        self.assertNotIn("generated_assistant", judge.seen[0])
        self.assertEqual(store.saved[0]["delivery_status"], "interrupted")
        self.assertEqual(store.saved[0]["generated_assistant"], "Theek hai, main note kar leti hoon.")

    def test_judge_sees_partial_handoff_separately_from_generated_answer(self):
        class Store:
            def __init__(self): self.saved = None
            def attach_followup(self, *_args): return []
            def append(self, *_args): pass
            def finalize_call(self, *_args): self.saved = _args[3]

        store = Store()
        session = TurnSession(build_graph(GenericRouter(), generic_agents(Chat())), conversation_store=store)
        session.ask_stream("Kal party hai", lambda _text: None, defer_delivery=True)
        session.mark_delivery("interrupted", handed_off_text="Theek hai,")
        session.finalize_call()
        self.assertEqual(store.saved[0]["assistant"], "Theek hai,")
        self.assertEqual(store.saved[0]["generated_assistant"], "Theek hai, main note kar leti hoon.")

    def test_order_window_reuses_same_order_and_refreshes_new_id(self):
        class Store:
            def __init__(self):
                self.requests = []

            def render_order_window(self, customer_id, query, active_order_id):
                self.requests.append((customer_id, query, active_order_id))
                return f"orders:{query or active_order_id or 'recent'}"

        store = Store()
        window = customer_order_window("caller-1", store)
        self.assertEqual(window("", {}), "orders:recent")
        self.assertEqual(window("What about delivery?", {}), "orders:recent")
        self.assertEqual(window("GRC-10003 status", {}), "orders:GRC-10003 status")
        self.assertEqual(window("When will it arrive?", {"order_id": "GRC-10003"}), "orders:GRC-10003 status")
        self.assertEqual(window("GRC-10004 status", {"order_id": "GRC-10003"}), "orders:GRC-10004 status")
        self.assertEqual(len(store.requests), 3)
        self.assertIsNone(store.requests[-1][2])

    def test_empty_grocery_window_does_not_pollute_a_personal_message(self):
        class Store:
            def __init__(self): self.calls = 0

            def render_order_window(self, *_args):
                self.calls += 1
                return "No account order records are available. Do not invent an order record."

        store = Store()
        window = customer_order_window("caller-1", store)
        self.assertEqual(window("", {}), "")
        self.assertEqual(window("Kal party hai", {}), "")
        self.assertIn("No account order records", window("Mera refund?", {}))
        self.assertEqual(store.calls, 1)

    def test_calendar_snapshot_is_reused_during_short_call(self):
        class Store:
            def __init__(self):
                self.calls = 0

            def snapshot(self, _owner_id):
                self.calls += 1
                return {"status": "busy", "busy_until": "2026-11-13T20:00:00+00:00"}

        store = Store()
        with patch.dict("os.environ", {"GOOGLE_CALENDAR_ENABLED": "true", "GOOGLE_CALENDAR_CALL_CACHE_SECONDS": "60"}), \
             patch("gnaniassignment.google_calendar.GoogleCalendarSettings.from_environment", return_value=SimpleNamespace(owner_id="owner")), \
             patch("gnaniassignment.google_calendar.GoogleCalendarStore", return_value=store):
            window = create_calendar_presence_window()
            self.assertEqual(window("", {}), window("Are you free?", {}))
        self.assertEqual(store.calls, 1)

    def test_knowledge_search_uses_query_terms_and_skips_empty_queries(self):
        self.assertEqual(_knowledge_search_query("Mera refund status kya hai?"), "refund OR status")

        class Cursor:
            def __init__(self):
                self.query = ""
                self.parameters = ()

            def __enter__(self): return self
            def __exit__(self, *_args): return False

            def execute(self, query, parameters):
                self.query, self.parameters = query, parameters

            def fetchall(self): return [("Refund policy",)]

        class Connection:
            def __init__(self, cursor): self._cursor = cursor
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def cursor(self): return self._cursor

        cursor = Cursor()
        module = SimpleNamespace(connect=lambda _url: Connection(cursor))
        store = PostgresKnowledgeStore("generic", "postgresql://unused")
        with patch.dict(sys.modules, {"psycopg": module}):
            self.assertEqual(store.search("Haan, please"), [])
            self.assertEqual(store.search("Refund status kya hai?"), ["Refund policy"])
        self.assertIn("websearch_to_tsquery", cursor.query)
        self.assertEqual(cursor.parameters, ("generic", "generic", "refund OR status", "refund OR status", 4))


if __name__ == "__main__":
    unittest.main()
