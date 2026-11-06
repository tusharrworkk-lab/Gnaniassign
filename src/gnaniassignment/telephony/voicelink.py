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
    started_at: float = field(default_factory=time.perf_counter)
    latency_store: CallLatencyStore | None = None

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
        async with self.send_lock:
            await self.websocket.send_json({"event": "media", "media": {"payload": _encode_alaw(converted)}})

    async def feed_media(self, payload: str) -> None:
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
        self.pcm_buffer.extend(pcm_16k)
        while len(self.pcm_buffer) >= self.stt.frame_bytes:
            frame = bytes(self.pcm_buffer[:self.stt.frame_bytes])
            del self.pcm_buffer[:self.stt.frame_bytes]
            await self.stt.send_pcm(frame)

    async def speak_reply(self, text: str) -> None:
        async for chunk in self.tts.synthesize(text):
            await self.send_media(chunk.audio)
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


async def handle_voicelink_websocket(websocket: WebSocket, events: Any | None = None) -> None:
    """Serve one documented VoiceLink WebSocket media session."""
    await websocket.accept()
    call: VoiceLinkCall | None = None
    receiver: asyncio.Task[None] | None = None
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
                call.latency_store = CallLatencyStore() if os.getenv("CALL_LATENCY_METRICS_ENABLED", "true").lower() == "true" else None
                await call.metric(call_started=True)
                logger.info("voicelink call started: stream=%s call=%s", stream_sid, call_sid)
                if events is not None:
                    await events.publish("call.started", session_id=stream_sid, caller_id=caller_id, call_id=call_sid, channel="voicelink")

                async def receive_stt() -> None:
                    assert call is not None
                    async for stt_event in call.stt.events():
                        if isinstance(stt_event, VoiceTranscript) and stt_event.text.strip():
                            await call.metric(stt_final_ms=round((time.perf_counter()-call.started_at)*1000,1), stt_provider_latency_ms=stt_event.latency_ms)
                            if hasattr(call.session, "record_stt_segment"):
                                call.session.record_stt_segment(stt_event.text, stt_event.segment_id, stt_event.latency_ms)
                            if events is not None:
                                await events.publish("transcript.received", session_id=call.stream_sid, customer_id=call.customer_id, text=stt_event.text)
                            result = await asyncio.to_thread(call.session.ask, stt_event.text)
                            await call.metric(llm_completed_ms=round((time.perf_counter()-call.started_at)*1000,1), llm_ttft_ms=result.get("metrics", {}).get("llm_ttft_ms"))
                            if events is not None:
                                await events.publish("agent.responded", session_id=call.stream_sid, intent=result["intent"], text=result["spoken"])
                            await call.speak_reply(result["spoken"])
                        elif isinstance(stt_event, VoiceError) and events is not None:
                            await events.publish("stt.failed", session_id=call.stream_sid, message=stt_event.message[:300])

                receiver = asyncio.create_task(receive_stt())
                continue
            if event_name == "media" and call is not None:
                media = event.get("media")
                if isinstance(media, dict) and media.get("track") == "inbound" and isinstance(media.get("payload"), str):
                    await call.feed_media(media["payload"])
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
        if call is not None:
            if hasattr(call.session, "finalize_call"):
                await asyncio.to_thread(call.session.finalize_call)
            await call.metric(call_ended=True, inbound_packets=call.inbound_packets)
            await call.tts.__aexit__(None, None, None)
            await call.stt.__aexit__(None, None, None)
            if events is not None:
                await events.publish("call.ended", session_id=call.stream_sid, call_id=call.call_sid, channel="voicelink")

