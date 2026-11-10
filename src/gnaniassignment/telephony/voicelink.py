"""VoiceLink WebSocket media adapter.

VoiceLink streams base64 A-law at 8 kHz. Gnani STT receives 16-bit PCM in
1024-byte frames; Gnani TTS returns linear PCM which we convert back to A-law.
The provider envelope is intentionally kept here, outside the agent core.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import binascii
import json
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from gnaniassignment.app.container import create_session
from gnaniassignment.stt.gnani import GnaniRealtimeStt
from gnaniassignment.tts.gnani import GnaniRealtimeTts, GnaniTtsError
from gnaniassignment.voice.events import VoiceError, VoiceTranscript
from gnaniassignment.voice.audio_enhancement import AudioEnhancer, create_audio_enhancer
from gnaniassignment.telephony.latency import CallLatencyStore


VOICE_LINK_SAMPLE_RATE = 8000
PCM_WIDTH = 2
VOICE_LINK_FRAME_BYTES = 160  # 20 ms: 8000 samples/s × 20 ms × 8-bit A-law
logger = logging.getLogger(__name__)


def _decode_alaw(payload: str) -> bytes:
    """Decode VoiceLink's documented base64 A-law media payload."""
    try:
        encoded = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("VoiceLink media.payload must be valid base64") from error
    return audioop.alaw2lin(encoded, PCM_WIDTH)


def _encode_alaw(pcm: bytes) -> str:
    """Return an outbound VoiceLink media.payload from 16-bit PCM."""
    return base64.b64encode(audioop.lin2alaw(pcm, PCM_WIDTH)).decode("ascii")


@dataclass
class VoiceLinkCall:
    """State that belongs to exactly one VoiceLink stream_sid."""

    websocket: WebSocket
    stream_sid: str
    call_sid: str
    caller_id: str
    customer_id: str
    stt: GnaniRealtimeStt
    tts: GnaniRealtimeTts
    session: Any
    input_resample_state: Any = None
    output_resample_state: Any = None
    pcm_buffer: bytearray = field(default_factory=bytearray)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inbound_packets: int = 0
    outbound_buffer: bytearray = field(default_factory=bytearray)
    outbound_next_frame_at: float | None = None
    outbound_last_frame_at: float | None = None
    started_at: float = field(default_factory=time.perf_counter)
    latency_store: CallLatencyStore | None = None
    reply_in_progress: bool = False
    vad: Any | None = None
    vad_mode: str = "off"
    vad_backend: str = "off"
    vad_speech_active: bool = False
    vad_turn_ended: asyncio.Event = field(default_factory=asyncio.Event)
    semantic_task: asyncio.Task[None] | None = None
    reply_count: int = 0
    _active_reply_metrics: dict[str, Any] | None = field(default=None, repr=False)
    turn_queue: asyncio.Queue[VoiceTranscript] | None = field(default=None, repr=False)
    latest_transcript: VoiceTranscript | None = None
    semantic_complete_pending: bool = False
    last_committed_transcript: tuple[str | None, str] | None = None
    events: Any | None = field(default=None, repr=False)
    audio_enhancer: AudioEnhancer | None = field(default=None, repr=False)
    enhancement_frames: int = 0
    turn_index: int = 0
    last_speech_started_ms: float | None = None
    last_speech_ended_ms: float | None = None
    last_turn_committed_ms: float | None = None
    active_turn_evaluation: dict[str, Any] | None = field(default=None, repr=False)
    reply_interrupted: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    barge_in_active: bool = False

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)

    async def begin_turn_evaluation(self) -> None:
        """Start a per-turn record. It deliberately contains no transcript/audio."""
        self.turn_index += 1
        started = self.elapsed_ms()
        speech_end = self.last_speech_ended_ms
        # VAD may be off in a controlled rollout. In that case provider-final
        # is the best available turn boundary, and the source is recorded so
        # dashboards never confuse it with a true acoustic end-of-speech.
        anchor = speech_end if speech_end is not None else (self.last_turn_committed_ms or started)
        self.active_turn_evaluation = {
            "turn_index": self.turn_index,
            "speech_started_at_ms": self.last_speech_started_ms,
            "speech_ended_at_ms": speech_end,
            "turn_committed_at_ms": self.last_turn_committed_ms or started,
            "response_latency_anchor_at_ms": anchor,
            "response_latency_anchor": "speech_end" if speech_end is not None else "turn_committed",
            "turn_processing_started_at_ms": started,
        }

    def note_llm_first_token(self) -> None:
        if self.active_turn_evaluation is None or "llm_ttft_latency_ms" in self.active_turn_evaluation:
            return
        now = self.elapsed_ms()
        self.active_turn_evaluation["llm_first_token_at_ms"] = now
        self.active_turn_evaluation["llm_ttft_latency_ms"] = round(now - self.active_turn_evaluation["turn_committed_at_ms"], 1)

    def note_first_outbound_audio(self) -> None:
        if self.active_turn_evaluation is None or "first_outbound_audio_at_ms" in self.active_turn_evaluation:
            return
        now = self.elapsed_ms()
        self.active_turn_evaluation["first_outbound_audio_at_ms"] = now
        self.active_turn_evaluation["first_outbound_audio_latency_ms"] = round(now - self.active_turn_evaluation["response_latency_anchor_at_ms"], 1)
        self.active_turn_evaluation["post_commit_first_outbound_audio_latency_ms"] = round(now - self.active_turn_evaluation["turn_committed_at_ms"], 1)

    def note_barge_in(self) -> bool:
        if self.active_turn_evaluation is None or "barge_in_detected_at_ms" in self.active_turn_evaluation:
            return False
        self.active_turn_evaluation["barge_in_detected_at_ms"] = self.elapsed_ms()
        self.active_turn_evaluation["barge_in_stop_supported"] = True
        return True

    async def interrupt_active_reply(self) -> bool:
        """Stop queued and future agent audio after a caller barge-in.

        VoiceLink's ``clear`` discards provider-side buffered media. The
        running TTS loop also observes ``reply_interrupted`` before emitting
        another frame, so this is a genuine hand-off stop rather than a metric
        that merely reports detection.
        """
        if not self.reply_in_progress or self.reply_interrupted.is_set():
            return False
        detected = self.note_barge_in()
        self.reply_interrupted.set()
        self.barge_in_active = True
        self.outbound_buffer.clear()
        async with self.send_lock:
            await self.websocket.send_json({"event": "clear", "stream_sid": self.stream_sid})
        stopped_at = self.elapsed_ms()
        if self.active_turn_evaluation is not None:
            self.active_turn_evaluation["barge_in_clear_sent_at_ms"] = stopped_at
            detected_at = self.active_turn_evaluation.get("barge_in_detected_at_ms")
            if isinstance(detected_at, (int, float)):
                self.active_turn_evaluation["barge_in_stop_latency_ms"] = round(stopped_at - detected_at, 1)
        await self.trace("barge_in.stopped", clear_sent=True, stop_latency_ms=(self.active_turn_evaluation or {}).get("barge_in_stop_latency_ms"))
        return detected

    async def finish_turn_evaluation(self, outcome: str) -> None:
        turn = self.active_turn_evaluation
        if turn is None:
            return
        turn["evaluation_finalized"] = True
        turn["outcome"] = outcome
        turn["finished_at_ms"] = self.elapsed_ms()
        silent_after_ms = max(100, int(os.getenv("VOICE_EVAL_SILENT_TURN_TIMEOUT_MS", "2500")))
        latency = turn.get("first_outbound_audio_latency_ms")
        turn["silent_turn"] = not isinstance(latency, (int, float)) or latency > silent_after_ms
        if self.latency_store is not None:
            await asyncio.to_thread(self.latency_store.record_turn, self.stream_sid, self.call_sid, self.customer_id, self.turn_index, dict(turn))
        await self.trace("eval.turn_completed", turn_index=self.turn_index, silent_turn=turn["silent_turn"], outcome=outcome, first_outbound_audio_latency_ms=latency)
        self.active_turn_evaluation = None

    async def metric(self, **values: Any) -> None:
        values["elapsed_ms"] = round((time.perf_counter() - self.started_at) * 1000, 1)
        if self.latency_store is not None:
            await asyncio.to_thread(self.latency_store.record, self.stream_sid, self.call_sid, self.customer_id, values)

    async def trace(self, kind: str, **values: Any) -> None:
        """Publish safe, real-time internals; never include raw audio/secrets."""
        if self.events is not None:
            await self.events.publish(
                kind,
                session_id=self.stream_sid,
                call_id=self.call_sid,
                customer_id=self.customer_id,
                channel="voicelink",
                **values,
            )

    async def send_media(self, pcm: bytes) -> None:
        """Resample linear TTS PCM and send the documented outbound frame."""
        if not pcm or self.reply_interrupted.is_set():
            return
        converted, self.output_resample_state = audioop.ratecv(
            pcm,
            PCM_WIDTH,
            1,
            self.tts.sample_rate,
            VOICE_LINK_SAMPLE_RATE,
            self.output_resample_state,
        )
        if not converted:
            return
        self.outbound_buffer.extend(audioop.lin2alaw(converted, PCM_WIDTH))
        while len(self.outbound_buffer) >= VOICE_LINK_FRAME_BYTES:
            frame = bytes(self.outbound_buffer[:VOICE_LINK_FRAME_BYTES])
            del self.outbound_buffer[:VOICE_LINK_FRAME_BYTES]
            await self._send_outbound_frame(frame)

    async def _send_outbound_frame(self, alaw: bytes) -> None:
        if self.reply_interrupted.is_set():
            return
        # VoiceLink expects telephony audio in 20 ms A-law packets. Pace from
        # a monotonic deadline instead of sleeping *after* send: that avoids
        # bursts when the event loop is delayed and gives us a real jitter
        # measurement for carrier-quality debugging.
        if len(alaw) != VOICE_LINK_FRAME_BYTES:
            raise ValueError("VoiceLink outbound frames must be exactly 160 bytes / 20 ms")
        if self.outbound_next_frame_at is not None:
            wait_seconds = self.outbound_next_frame_at - time.perf_counter()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
        async with self.send_lock:
            await self.websocket.send_json({
                "event": "media", "stream_sid": self.stream_sid,
                "media": {"payload": base64.b64encode(alaw).decode("ascii")},
            })
        sent_at = time.perf_counter()
        self.outbound_next_frame_at = sent_at + 0.02
        # This measures that our server handed a frame to the WebSocket.  It
        # is deliberately *not* labelled "played": VoiceLink's documented
        # media/mark events do not acknowledge handset playback.
        if self._active_reply_metrics is not None:
            now_ms = self.elapsed_ms()
            if self.outbound_last_frame_at is not None:
                gap_ms = round((sent_at - self.outbound_last_frame_at) * 1000, 1)
                self._active_reply_metrics["outbound_max_frame_gap_ms"] = max(
                    float(self._active_reply_metrics.get("outbound_max_frame_gap_ms", 0)), gap_ms
                )
            self.outbound_last_frame_at = sent_at
            self._active_reply_metrics["outbound_frames_sent"] += 1
            self._active_reply_metrics["outbound_last_frame_sent_ms"] = now_ms
            self._active_reply_metrics.setdefault("outbound_first_frame_sent_ms", now_ms)
            if self._active_reply_metrics["outbound_frames_sent"] == 1:
                self.note_first_outbound_audio()
                await self.trace("tts.outbound_first_frame", reply_index=self.reply_count, elapsed_ms=now_ms)

    async def flush_outbound(self) -> None:
        if self.outbound_buffer and not self.reply_interrupted.is_set():
            # Never send a short terminal packet. A-law telephony gateways
            # consume fixed 20 ms frames; zero-padding is silence and avoids
            # a truncated/corrupted tail at sentence boundaries.
            frame = bytes(self.outbound_buffer).ljust(VOICE_LINK_FRAME_BYTES, b"\xd5")
            self.outbound_buffer.clear()
            await self._send_outbound_frame(frame)

    async def feed_media(self, payload: str) -> list[Any]:
        """Decode, resample, and frame inbound VoiceLink audio for Gnani."""
        self.inbound_packets += 1
        if self.inbound_packets == 1:
            await self.metric(media_first_packet_ms=round((time.perf_counter()-self.started_at)*1000,1))
            await self.trace("media.inbound_started", sample_rate=VOICE_LINK_SAMPLE_RATE, encoding="audio/alaw")
        if self.inbound_packets == 1 or self.inbound_packets % 100 == 0:
            logger.info("voicelink inbound media: stream=%s packets=%d", self.stream_sid, self.inbound_packets)
        pcm_8k = _decode_alaw(payload)
        pcm_16k, self.input_resample_state = audioop.ratecv(
            pcm_8k,
            PCM_WIDTH,
            1,
            VOICE_LINK_SAMPLE_RATE,
            self.stt.sample_rate,
            self.input_resample_state,
        )
        if self.audio_enhancer is not None:
            enhanced = self.audio_enhancer.process_pcm_s16le(pcm_16k)
            pcm_16k = enhanced.audio
            self.enhancement_frames += 1
            # One event per second at VoiceLink's 20 ms cadence keeps the
            # debugging feed useful without flooding it with audio metadata.
            if self.enhancement_frames == 1 or self.enhancement_frames % 50 == 0:
                await self.metric(
                    audio_enhancement_mode=self.audio_enhancer.mode,
                    audio_enhancement_backend=self.audio_enhancer.backend,
                    audio_enhancement_input_rms_dbfs=enhanced.input_rms_dbfs,
                    audio_enhancement_output_rms_dbfs=enhanced.output_rms_dbfs,
                    audio_enhancement_processing_ms=enhanced.processing_ms,
                )
                await self.trace(
                    "audio.enhancement",
                    mode=self.audio_enhancer.mode,
                    backend=self.audio_enhancer.backend,
                    input_rms_dbfs=enhanced.input_rms_dbfs,
                    output_rms_dbfs=enhanced.output_rms_dbfs,
                    speech_probability=enhanced.speech_probability,
                    processing_ms=enhanced.processing_ms,
                )
        boundaries: list[Any] = []
        if self.vad is not None:
            # Both optional detectors consume the exact 16 kHz PCM stream
            # sent to STT. TEN requires it; Silero also supports it.
            boundaries = self.vad.feed_pcm_s16le(pcm_16k)
            for boundary in boundaries:
                if boundary.kind == "speech_started":
                    self.last_speech_started_ms = self.elapsed_ms()
                    self.vad_speech_active = True
                    self.vad_turn_ended.clear()
                    self.latest_transcript = None
                    self.semantic_complete_pending = False
                    self.last_committed_transcript = None
                elif boundary.kind == "speech_ended":
                    self.last_speech_ended_ms = self.elapsed_ms()
                    self.vad_speech_active = False
                    # TEN's acoustic pause is only a trigger for semantic
                    # analysis. It is not yet permission to commit a turn.
                    if self.vad_backend != "ten_pipecat":
                        self.vad_turn_ended.set()
        self.pcm_buffer.extend(pcm_16k)
        while len(self.pcm_buffer) >= self.stt.frame_bytes:
            frame = bytes(self.pcm_buffer[:self.stt.frame_bytes])
            del self.pcm_buffer[:self.stt.frame_bytes]
            await self.stt.send_pcm(frame)
        return boundaries

    async def evaluate_semantic_pause(self, events: Any | None) -> None:
        """Let Smart Turn decide whether a TEN-detected pause ends the turn."""
        detector = self.vad
        if detector is None or not hasattr(detector, "analyze_current_pause"):
            return
        decision = await detector.analyze_current_pause(detector.revision)
        if decision.stale:
            return
        await self.metric(
            smart_turn_complete=decision.complete,
            smart_turn_probability=decision.probability,
        )
        if events is not None:
            await events.publish(
                "smart_turn.complete" if decision.complete else "smart_turn.incomplete",
                session_id=self.stream_sid,
                channel="voicelink",
                probability=decision.probability,
            )
        if decision.complete:
            self.vad_turn_ended.set()
            self.semantic_complete_pending = True
            await self.commit_latest_transcript("smart_turn")

    async def commit_latest_transcript(self, reason: str) -> bool:
        """Promote the newest STT hypothesis exactly once for a completed turn.

        Smart Turn can now commit an explicit non-final STT hypothesis after a
        semantic speech end. If Gnani emits only final transcripts, behavior
        stays unchanged and the provider final is committed as before.
        """
        transcript = self.latest_transcript
        if transcript is None or not transcript.text.strip() or self.turn_queue is None:
            return False
        signature = (transcript.segment_id, transcript.text.strip())
        if signature == self.last_committed_transcript:
            return False
        self.last_committed_transcript = signature
        self.last_turn_committed_ms = self.elapsed_ms()
        await self.turn_queue.put(transcript)
        await self.metric(stt_turn_committed_by=reason)
        await self.trace("transcript.committed", text=transcript.text, final=transcript.is_final, reason=reason)
        logger.info("voicelink transcript committed: stream=%s reason=%s segment=%s", self.stream_sid, reason, transcript.segment_id)
        return True

    async def speak_reply(self, text: str, *, add_clear_throat: bool = False) -> None:
        """Synthesize and send one streamed reply with auditable hand-off timing.

        VoiceLink documents outbound media as fire-and-forget.  These values
        therefore isolate TTS and server-to-VoiceLink hand-off, while making
        no unsupported claim about when the caller's handset played the audio.
        """
        if self.reply_interrupted.is_set():
            return
        self.reply_count += 1
        # Each TTS request is independent speech separated by silence. Carrying
        # resampler filter state across replies can create a click/glitch at a
        # new response boundary, especially when a provider changes chunking.
        self.output_resample_state = None
        self.outbound_next_frame_at = None
        self.outbound_last_frame_at = None
        reply_started = time.perf_counter()
        self._active_reply_metrics = {
            "reply_index": self.reply_count,
            "reply_text_characters": len(text),
            "reply_started_ms": round((reply_started - self.started_at) * 1000, 1),
            "outbound_frames_sent": 0,
            "clear_throat_inserted": add_clear_throat,
        }
        try:
            await self.trace(
                "tts.started", reply_index=self.reply_count, text_characters=len(text),
                clear_throat_inserted=add_clear_throat,
            )
            if add_clear_throat:
                await self.trace("tts.clear_throat", reply_index=self.reply_count)
            for attempt in range(2):
                audio_chunks = 0
                async for chunk in self.tts.synthesize(text, add_clear_throat=add_clear_throat):
                    if self.reply_interrupted.is_set():
                        self.outbound_buffer.clear()
                        return
                    audio_chunks += 1
                    if self.tts.last_first_audio_ms is not None:
                        self._active_reply_metrics.setdefault(
                            "tts_first_audio_ms",
                            round((reply_started - self.started_at) * 1000 + self.tts.last_first_audio_ms, 1),
                        )
                        if not self._active_reply_metrics.get("tts_first_audio_emitted"):
                            self._active_reply_metrics["tts_first_audio_emitted"] = True
                            await self.trace("tts.first_audio", reply_index=self.reply_count, tts_ttfa_ms=round(self.tts.last_first_audio_ms, 1))
                    await self.send_media(chunk.audio)
                if audio_chunks:
                    break
                if attempt == 0:
                    # A terminal complete event with zero audio is not a
                    # successful spoken reply. Reopen and retry once: no
                    # caller audio has been sent, so duplication is impossible.
                    await self.trace("tts.zero_audio_retry", reply_index=self.reply_count)
                    await self.tts.__aexit__(None, None, None)
                    await self.tts.__aenter__()
                    continue
                raise GnaniTtsError("Gnani TTS completed without producing audio")
            await self.flush_outbound()
            if self.reply_interrupted.is_set():
                return
            self._active_reply_metrics["tts_completed_ms"] = round(
                (reply_started - self.started_at) * 1000 + (self.tts.last_synthesis_ms or 0), 1
            )
            # This outbound marker is optional for playback, but lets VoiceLink
            # correlate an agent response boundary using its documented schema.
            async with self.send_lock:
                await self.websocket.send_json({
                    "event": "mark", "stream_sid": self.stream_sid,
                    "mark": {"name": "response_done"},
                })
            self._active_reply_metrics["outbound_mark_sent_ms"] = round((time.perf_counter() - self.started_at) * 1000, 1)
            await self.metric(reply_count=self.reply_count, latest_reply=dict(self._active_reply_metrics))
            await self.trace("tts.completed", reply_index=self.reply_count, frames=self._active_reply_metrics["outbound_frames_sent"])
            logger.info("voicelink reply handed off: stream=%s metrics=%s", self.stream_sid, self._active_reply_metrics)
        finally:
            self._active_reply_metrics = None


def _call_context(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    start = payload.get("start")
    if not isinstance(start, dict):
        raise ValueError("VoiceLink start event requires a start object")
    stream_sid = str(start.get("stream_sid") or payload.get("stream_sid") or "").strip()
    call_sid = str(start.get("call_sid") or "").strip()
    caller_id = str(start.get("from") or "unknown").strip()
    if not stream_sid or not call_sid:
        raise ValueError("VoiceLink start event requires stream_sid and call_sid")
    media_format = start.get("media_format")
    if not isinstance(media_format, dict) or media_format.get("encoding") != "audio/alaw" or str(media_format.get("sample_rate")) != "8000":
        raise ValueError("VoiceLink stream must use documented audio/alaw at 8000 Hz")
    custom = start.get("custom_parameters")
    customer_id = custom.get("customer_id") if isinstance(custom, dict) else None
    return stream_sid, call_sid, caller_id, str(customer_id or f"caller:{caller_id}")


async def _reply_to_transcript(call: VoiceLinkCall, stt_event: VoiceTranscript, events: Any | None) -> None:
    """Generate one response without letting a recoverable model failure kill
    the STT receiver for the rest of the telephone call.
    """
    outcome = "completed"
    call.reply_interrupted.clear()
    call.barge_in_active = False
    await call.begin_turn_evaluation()
    try:
        await call.metric(stt_final_ms=round((time.perf_counter()-call.started_at)*1000,1), stt_provider_latency_ms=stt_event.latency_ms)
        llm_started = time.perf_counter()
        await call.trace("llm.started", transcript_characters=len(stt_event.text), transcript_final=stt_event.is_final)
        if hasattr(call.session, "record_stt_segment"):
            call.session.record_stt_segment(stt_event.text, stt_event.segment_id, stt_event.latency_ms)
        if events is not None:
            await events.publish("transcript.received", session_id=call.stream_sid, customer_id=call.customer_id, text=stt_event.text)
        chunks: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        first_chunk_at: float | None = None
        clear_throat_started = False

        async def play_proactive_clear_throat() -> bool:
            """Play a tiny standalone cue only while the answer is pending.

            This is intentionally a timer, not a decision made after the
            first answer text arrives.  The caller hears the cue at the
            configured delay, then the normal answer is synthesized unchanged.
            """
            nonlocal clear_throat_started
            if os.getenv("GNANI_TTS_POST_RESPONSE_CLEAR_THROAT_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}:
                return False
            threshold_ms = max(0.0, float(os.getenv("GNANI_TTS_POST_RESPONSE_CLEAR_THROAT_MS", "600")))
            await asyncio.sleep(threshold_ms / 1000)
            if first_chunk_at is not None or generation.done() or call.reply_interrupted.is_set():
                return False
            clear_throat_started = True
            await call.trace("tts.clear_throat", proactive=True, threshold_ms=threshold_ms)
            await call.speak_reply("", add_clear_throat=True)
            return True

        def on_chunk(chunk: str) -> None:
            loop.call_soon_threadsafe(chunks.put_nowait, chunk)

        generation = asyncio.create_task(asyncio.to_thread(call.session.ask_stream, stt_event.text, on_chunk))
        clear_throat_task = asyncio.create_task(play_proactive_clear_throat())
        pending = ""
        try:
            while not generation.done() or not chunks.empty():
                try:
                    chunk = await asyncio.wait_for(chunks.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    call.note_llm_first_token()
                    await call.metric(llm_ttft_ms=round((first_chunk_at-call.started_at)*1000,1))
                    await call.trace("llm.first_token", ttft_ms=round((first_chunk_at - llm_started) * 1000, 1))
                    # If the timer is still asleep it must never play later.
                    # If it already started, finish the short cue before
                    # sending words, preserving a clean telephony stream.
                    if not clear_throat_started and not clear_throat_task.done():
                        clear_throat_task.cancel()
                    try:
                        await clear_throat_task
                    except asyncio.CancelledError:
                        pass
                pending += chunk
                # Start audio at a natural sentence boundary, or cap the fragment
                # so a slow model never becomes a voice note.
                if any(mark in pending for mark in ".!?") or len(pending) >= 90:
                    await call.speak_reply(pending.strip())
                    pending = ""
                    if call.reply_interrupted.is_set():
                        outcome = "interrupted"
                        break
            result = await generation
            if pending.strip() and not call.reply_interrupted.is_set():
                await call.speak_reply(pending.strip())
        finally:
            if not clear_throat_task.done():
                clear_throat_task.cancel()
            try:
                await clear_throat_task
            except asyncio.CancelledError:
                pass
        if call.reply_interrupted.is_set():
            outcome = "interrupted"
        await call.metric(llm_completed_ms=round((time.perf_counter()-call.started_at)*1000,1))
        await call.trace("llm.completed", intent=result["intent"], response_characters=len(result["spoken"] or ""))
        if events is not None:
            await events.publish("agent.responded", session_id=call.stream_sid, intent=result["intent"], text=result["spoken"])
    except Exception as error:
        outcome = "failed"
        # A model may exhaust its reasoning/output budget. The caller should
        # hear a retry prompt and the next final STT segment must still work.
        logger.exception("voicelink turn failed: stream=%s", call.stream_sid)
        await call.metric(agent_failed=True, agent_error=type(error).__name__)
        if events is not None:
            await events.publish("agent.failed", session_id=call.stream_sid, channel="voicelink", message=str(error)[:300])
        try:
            await call.speak_reply("Maaf kijiye, main jawab taiyar nahi kar paayi. Kripya ek baar phir se bataiye.")
        except Exception:
            logger.exception("voicelink fallback speech failed: stream=%s", call.stream_sid)
    finally:
        await call.finish_turn_evaluation(outcome)


async def handle_voicelink_websocket(websocket: WebSocket, events: Any | None = None) -> None:
    """Serve one documented VoiceLink WebSocket media session."""
    await websocket.accept()
    call: VoiceLinkCall | None = None
    receiver: asyncio.Task[None] | None = None
    turn_worker: asyncio.Task[None] | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"event": "clear", "stream_sid": call.stream_sid if call else ""})
                continue
            if not isinstance(event, dict):
                continue
            event_name = event.get("event")
            logger.info("voicelink event received: event=%s stream=%s", event_name, call.stream_sid if call else event.get("stream_sid", "pending"))
            if event_name == "connected":
                continue
            if event_name == "start":
                if call is not None:
                    raise ValueError("VoiceLink stream sent start more than once")
                stream_sid, call_sid, caller_id, customer_id = _call_context(event)
                # VoiceLink endpointing owns the completed-turn decision.
                # Request a short provider final-silence window so an STT
                # final arrives promptly even when no interim hypothesis is
                # available to Smart Turn.
                stt = GnaniRealtimeStt(
                    min_silence_ms=int(os.getenv("VOICE_LINK_STT_MIN_SILENCE_MS", "350")),
                )
                tts = GnaniRealtimeTts()
                await stt.__aenter__()
                await tts.__aenter__()
                call = VoiceLinkCall(websocket, stream_sid, call_sid, caller_id, customer_id, stt, tts, create_session(customer_id=customer_id, session_id=stream_sid))
                call.events = events
                call.audio_enhancer = create_audio_enhancer(call.stt.sample_rate)
                # New generic settings supersede the legacy Silero-only
                # switch. Keeping that switch preserves existing deployments.
                vad_mode = os.getenv(
                    "VOICE_LINK_TURN_DETECTION_MODE",
                    os.getenv("VOICE_LINK_SILERO_VAD_MODE", "off"),
                ).lower()
                if vad_mode not in {"off", "shadow", "endpoint"}:
                    raise ValueError("VOICE_LINK_TURN_DETECTION_MODE must be off, shadow, or endpoint")
                if vad_mode != "off":
                    backend = os.getenv("VOICE_LINK_TURN_DETECTOR", "ten_pipecat").lower()
                    if backend == "ten_pipecat":
                        from gnaniassignment.voice.ten_smart_turn import TenPipecatTurnDetector
                        call.vad = TenPipecatTurnDetector(
                            threshold=float(os.getenv("VOICE_LINK_TEN_VAD_THRESHOLD", "0.5")),
                            hop_samples=int(os.getenv("VOICE_LINK_TEN_VAD_HOP_SAMPLES", "256")),
                            min_speech_ms=int(os.getenv("VOICE_LINK_TEN_VAD_MIN_SPEECH_MS", "160")),
                            min_silence_ms=int(os.getenv("VOICE_LINK_TEN_VAD_MIN_SILENCE_MS", "320")),
                        )
                    elif backend == "silero":
                        from gnaniassignment.voice.silero_vad import SileroVad
                        call.vad = SileroVad(
                            call.stt.sample_rate,
                            threshold=float(os.getenv("VOICE_LINK_SILERO_VAD_THRESHOLD", "0.55")),
                            min_silence_ms=int(os.getenv("VOICE_LINK_SILERO_VAD_MIN_SILENCE_MS", "350")),
                        )
                    else:
                        raise ValueError("VOICE_LINK_TURN_DETECTOR must be ten_pipecat or silero")
                    call.vad_mode = vad_mode
                    call.vad_backend = backend
                call.latency_store = CallLatencyStore() if os.getenv("CALL_LATENCY_METRICS_ENABLED", "true").lower() == "true" else None
                await call.metric(call_started=True)
                logger.info("voicelink call started: stream=%s call=%s", stream_sid, call_sid)
                if events is not None:
                    await events.publish("call.started", session_id=stream_sid, caller_id=caller_id, call_id=call_sid, channel="voicelink")

                # A telephone speaker naturally pauses inside one sentence.
                # Gnani can deliver each pause as a final transcript, so route
                # a short cluster as one user turn instead of asking the LLM
                # to respond to every fragment.
                coalesce_ms = max(250, min(int(os.getenv("VOICELINK_TURN_COALESCE_MS", "1100")), 3000))
                turns: asyncio.Queue[VoiceTranscript] = asyncio.Queue()
                call.turn_queue = turns

                async def process_turns() -> None:
                    assert call is not None
                    while True:
                        first = await turns.get()
                        segments = [first]
                        while True:
                            # Wait for either a new STT segment or an endpoint
                            # decision. Smart Turn's result must wake this
                            # worker; otherwise a semantic completion would
                            # still pay the legacy coalescing timeout.
                            wait_seconds = coalesce_ms / 1000
                            get_segment = asyncio.create_task(turns.get())
                            endpoint_wait: asyncio.Task[bool] | None = None
                            waiters: set[asyncio.Task[Any]] = {get_segment}
                            if call.vad_mode == "endpoint":
                                endpoint_wait = asyncio.create_task(call.vad_turn_ended.wait())
                                waiters.add(endpoint_wait)
                            done, pending = await asyncio.wait(
                                waiters,
                                timeout=wait_seconds,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for waiter in pending:
                                waiter.cancel()
                            if not done:
                                break
                            if get_segment in done:
                                segments.append(get_segment.result())
                                continue
                            # A semantic completion gets a tiny grace window
                            # for any STT final already in flight, then commits.
                            try:
                                segments.append(await asyncio.wait_for(turns.get(), timeout=float(os.getenv("VOICE_LINK_VAD_ENDPOINT_GRACE_MS", "250")) / 1000))
                            except asyncio.TimeoutError:
                                pass
                            break
                        combined = VoiceTranscript(
                            text=" ".join(segment.text.strip() for segment in segments if segment.text.strip()),
                            segment_id=segments[-1].segment_id,
                            segment_index=segments[-1].segment_index,
                            audio_duration_ms=sum(segment.audio_duration_ms or 0 for segment in segments) or None,
                            latency_ms=segments[-1].latency_ms,
                        )
                        if not combined.text:
                            continue
                        call.reply_in_progress = True
                        try:
                            await _reply_to_transcript(call, combined, events)
                        finally:
                            call.reply_in_progress = False

                async def receive_stt() -> None:
                    assert call is not None
                    async for stt_event in call.stt.events():
                        if isinstance(stt_event, VoiceTranscript) and stt_event.text.strip():
                            call.latest_transcript = stt_event
                            await call.trace(
                                "transcript.final" if stt_event.is_final else "transcript.partial",
                                text=stt_event.text,
                                segment_id=stt_event.segment_id,
                                provider_latency_ms=stt_event.latency_ms,
                            )
                            if call.reply_in_progress and not call.barge_in_active:
                                logger.info("voicelink transcript discarded while replying: stream=%s", call.stream_sid)
                                if events is not None:
                                    await events.publish("transcript.discarded", session_id=call.stream_sid, channel="voicelink", reason="agent_reply_in_progress")
                                continue
                            if stt_event.is_final:
                                await call.commit_latest_transcript("provider_final")
                            elif call.semantic_complete_pending:
                                await call.commit_latest_transcript("smart_turn")
                            elif events is not None:
                                await events.publish("transcript.partial", session_id=call.stream_sid, customer_id=call.customer_id, text=stt_event.text)
                        elif isinstance(stt_event, VoiceError) and events is not None:
                            await events.publish("stt.failed", session_id=call.stream_sid, message=stt_event.message[:300])

                turn_worker = asyncio.create_task(process_turns())
                receiver = asyncio.create_task(receive_stt())
                continue
            if event_name == "media" and call is not None:
                media = event.get("media")
                if isinstance(media, dict) and media.get("track") == "inbound" and isinstance(media.get("payload"), str):
                    boundaries = await call.feed_media(media["payload"])
                    for boundary in boundaries:
                        if boundary.kind == "speech_started" and call.reply_in_progress:
                            stopped = await call.interrupt_active_reply()
                            if stopped:
                                await call.trace("barge_in.detected", detector=call.vad_backend, stop_supported=True)
                        await call.metric(turn_detector_event=boundary.kind, turn_detector=call.vad_backend)
                        if events is not None:
                            await events.publish("vad." + boundary.kind, session_id=call.stream_sid, channel="voicelink", mode=call.vad_mode, detector=call.vad_backend)
                        if boundary.kind == "speech_ended" and call.vad_backend == "ten_pipecat":
                            if call.semantic_task is not None and not call.semantic_task.done():
                                call.semantic_task.cancel()
                            call.semantic_task = asyncio.create_task(call.evaluate_semantic_pause(events))
                continue
            if event_name == "stop":
                logger.info("voicelink call stop: stream=%s packets=%d", call.stream_sid if call else "unknown", call.inbound_packets if call else 0)
                break
    except Exception as error:
        logger.exception("voicelink call failed: stream=%s", call.stream_sid if call else "unknown")
        if events is not None:
            await events.publish("call.failed", session_id=call.stream_sid if call else None, channel="voicelink", message=str(error)[:300])
        raise
    finally:
        if receiver is not None:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
        if turn_worker is not None:
            turn_worker.cancel()
            try:
                await turn_worker
            except asyncio.CancelledError:
                pass
        if call is not None:
            if call.semantic_task is not None:
                call.semantic_task.cancel()
                try:
                    await call.semantic_task
                except asyncio.CancelledError:
                    pass
            if call.vad is not None and hasattr(call.vad, "close"):
                await call.vad.close()
            if hasattr(call.session, "finalize_call"):
                await asyncio.to_thread(call.session.finalize_call)
            await call.metric(call_ended=True, inbound_packets=call.inbound_packets)
            await call.tts.__aexit__(None, None, None)
            await call.stt.__aexit__(None, None, None)
            if events is not None:
                await events.publish("call.ended", session_id=call.stream_sid, call_id=call.call_sid, channel="voicelink")

