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
from gnaniassignment.tts.gnani import GnaniRealtimeTts
from gnaniassignment.voice.events import VoiceError, VoiceTranscript
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
    started_at: float = field(default_factory=time.perf_counter)
    latency_store: CallLatencyStore | None = None
    reply_in_progress: bool = False
    vad: Any | None = None
    vad_mode: str = "off"
    vad_backend: str = "off"
    vad_speech_active: bool = False
    vad_turn_ended: asyncio.Event = field(default_factory=asyncio.Event)
    semantic_task: asyncio.Task[None] | None = None

    async def metric(self, **values: Any) -> None:
        values["elapsed_ms"] = round((time.perf_counter() - self.started_at) * 1000, 1)
        if self.latency_store is not None:
            await asyncio.to_thread(self.latency_store.record, self.stream_sid, self.call_sid, self.customer_id, values)

    async def send_media(self, pcm: bytes) -> None:
        """Resample linear TTS PCM and send the documented outbound frame."""
        if not pcm:
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
        async with self.send_lock:
            await self.websocket.send_json({"event": "media", "media": {"payload": base64.b64encode(alaw).decode("ascii")}})
        # Preserve real-time audio cadence; sending a full TTS response as fast
        # as the network allows causes carrier-side jitter/garbling.
        await asyncio.sleep(0.02)

    async def flush_outbound(self) -> None:
        if self.outbound_buffer:
            frame = bytes(self.outbound_buffer)
            self.outbound_buffer.clear()
            await self._send_outbound_frame(frame)

    async def feed_media(self, payload: str) -> list[Any]:
        """Decode, resample, and frame inbound VoiceLink audio for Gnani."""
        self.inbound_packets += 1
        if self.inbound_packets == 1: await self.metric(media_first_packet_ms=round((time.perf_counter()-self.started_at)*1000,1))
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
        boundaries: list[Any] = []
        if self.vad is not None:
            # Both optional detectors consume the exact 16 kHz PCM stream
            # sent to STT. TEN requires it; Silero also supports it.
            boundaries = self.vad.feed_pcm_s16le(pcm_16k)
            for boundary in boundaries:
                if boundary.kind == "speech_started":
                    self.vad_speech_active = True
                    self.vad_turn_ended.clear()
                elif boundary.kind == "speech_ended":
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

    async def speak_reply(self, text: str) -> None:
        async for chunk in self.tts.synthesize(text):
            await self.send_media(chunk.audio)
        await self.flush_outbound()
        # This outbound marker is optional for playback, but lets VoiceLink
        # correlate an agent response boundary using its documented schema.
        async with self.send_lock:
            await self.websocket.send_json({"event": "mark", "mark": {"name": "response_done"}})


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
    try:
        await call.metric(stt_final_ms=round((time.perf_counter()-call.started_at)*1000,1), stt_provider_latency_ms=stt_event.latency_ms)
        if hasattr(call.session, "record_stt_segment"):
            call.session.record_stt_segment(stt_event.text, stt_event.segment_id, stt_event.latency_ms)
        if events is not None:
            await events.publish("transcript.received", session_id=call.stream_sid, customer_id=call.customer_id, text=stt_event.text)
        chunks: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        first_chunk_at: float | None = None

        def on_chunk(chunk: str) -> None:
            loop.call_soon_threadsafe(chunks.put_nowait, chunk)

        generation = asyncio.create_task(asyncio.to_thread(call.session.ask_stream, stt_event.text, on_chunk))
        pending = ""
        while not generation.done() or not chunks.empty():
            try:
                chunk = await asyncio.wait_for(chunks.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                await call.metric(llm_ttft_ms=round((first_chunk_at-call.started_at)*1000,1))
            pending += chunk
            # Start audio at a natural sentence boundary, or cap the fragment
            # so a slow model never becomes a voice note.
            if any(mark in pending for mark in ".!?") or len(pending) >= 90:
                await call.speak_reply(pending.strip())
                pending = ""
        result = await generation
        if pending.strip():
            await call.speak_reply(pending.strip())
        await call.metric(llm_completed_ms=round((time.perf_counter()-call.started_at)*1000,1))
        if events is not None:
            await events.publish("agent.responded", session_id=call.stream_sid, intent=result["intent"], text=result["spoken"])
    except Exception as error:
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
                stt, tts = GnaniRealtimeStt(), GnaniRealtimeTts()
                await stt.__aenter__()
                await tts.__aenter__()
                call = VoiceLinkCall(websocket, stream_sid, call_sid, caller_id, customer_id, stt, tts, create_session(customer_id=customer_id))
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
                            if call.reply_in_progress:
                                logger.info("voicelink transcript discarded while replying: stream=%s", call.stream_sid)
                                if events is not None:
                                    await events.publish("transcript.discarded", session_id=call.stream_sid, channel="voicelink", reason="agent_reply_in_progress")
                                continue
                            await turns.put(stt_event)
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

