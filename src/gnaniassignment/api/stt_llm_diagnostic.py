"""Authenticated production STT -> LLM diagnostic WebSocket.

The socket accepts raw 16-bit mono PCM frames from a trusted developer client
and returns transcripts, streamed answer text, and server-side latency data.
It deliberately has no TTS or telephony adapter in its path.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from functools import partial
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from gnaniassignment.app.container import create_session
from gnaniassignment.voice.events import VoiceError, VoiceProcessing, VoiceTranscript
from gnaniassignment.voice.providers import create_stt_provider
from gnaniassignment.voice.turn_bridge import TurnBridge


def _number(value: object) -> float | None:
    return round(float(value), 1) if isinstance(value, (int, float)) else None


async def handle_stt_llm_diagnostic_websocket(websocket: WebSocket) -> None:
    """Run one isolated production-composition diagnostic session."""
    await websocket.accept()
    try:
        raw_start = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        start = json.loads(raw_start)
    except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        await websocket.close(code=1003, reason="Send a valid start message first")
        return
    if start.get("type") != "start":
        await websocket.close(code=1003, reason="First message must have type=start")
        return

    customer_id = str(start.get("customer_id") or "diagnostic")[:256]
    requested_persistence = start.get("persist") is True
    persistence_allowed = os.getenv(
        "STT_LLM_DIAGNOSTIC_ALLOW_PERSISTENCE", "false"
    ).lower() in {"1", "true", "yes", "on"}
    persist = requested_persistence and persistence_allowed
    session_id = f"diagnostic:{uuid4()}"
    outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
    loop = asyncio.get_running_loop()
    started_at = time.perf_counter()
    measurements: list[dict[str, Any]] = []

    async def send_events() -> None:
        while True:
            event = await outgoing.get()
            try:
                await websocket.send_text(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            finally:
                outgoing.task_done()

    def emit(event_type: str, **payload: Any) -> None:
        event = {
            "type": event_type,
            "at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            **payload,
        }
        try:
            outgoing.put_nowait(event)
        except asyncio.QueueFull:
            # Losing a diagnostic event is preferable to stalling live audio.
            pass

    def emit_from_thread(event_type: str, **payload: Any) -> None:
        loop.call_soon_threadsafe(partial(emit, event_type, **payload))

    sender = asyncio.create_task(send_events())
    try:
        composition_started = time.perf_counter()
        session = await asyncio.to_thread(
            create_session,
            customer_id=customer_id,
            session_id=session_id,
            persist_conversation=persist,
        )
        composition_ms = (time.perf_counter() - composition_started) * 1000
        bridge = TurnBridge(session)
        stt = create_stt_provider()
        emit(
            "connected",
            customer_id=customer_id,
            persisted=persist,
            composition_ms=round(composition_ms, 1),
            stt={
                "provider": type(stt).__name__,
                "model": getattr(stt, "model", None),
                "language_code": getattr(stt, "language_code", None),
                "sample_rate": stt.sample_rate,
                "frame_bytes": stt.frame_bytes,
                "min_silence_ms": stt.min_silence_ms,
            },
        )

        stt_connect_started = time.perf_counter()
        async with stt:
            emit("stt.connected", connect_ms=round((time.perf_counter() - stt_connect_started) * 1000, 1))
            processing_at: float | None = None

            async def receive_audio() -> str:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return "disconnect"
                    frame = message.get("bytes")
                    if isinstance(frame, bytes):
                        if len(frame) != stt.frame_bytes:
                            emit("input.error", message=f"Expected {stt.frame_bytes} PCM bytes, received {len(frame)}")
                            continue
                        await stt.send_pcm(frame)
                        continue
                    text = message.get("text")
                    if not isinstance(text, str):
                        continue
                    try:
                        control = json.loads(text)
                    except json.JSONDecodeError:
                        emit("input.error", message="Control messages must be JSON")
                        continue
                    if control.get("type") == "stop":
                        return "stop"

            async def process_stt_events() -> None:
                nonlocal processing_at
                async for event in stt.events():
                    if isinstance(event, VoiceProcessing):
                        processing_at = time.perf_counter()
                        emit("stt.processing")
                        continue
                    if isinstance(event, VoiceError):
                        emit("stt.error", message=event.message)
                        continue
                    if not isinstance(event, VoiceTranscript):
                        continue
                    if not event.is_final:
                        emit("transcript.partial", text=event.text)
                        continue
                    if not event.text.strip():
                        continue

                    committed_at = time.perf_counter()
                    endpoint_to_final_ms = (
                        (committed_at - processing_at) * 1000 if processing_at is not None else None
                    )
                    processing_at = None
                    turn_index = len(measurements) + 1
                    session.record_stt_segment(
                        event.text,
                        event.segment_id,
                        event.latency_ms,
                    )
                    detected_language_code = getattr(
                        event, "detected_language_code", None
                    )
                    emit(
                        "transcript.final",
                        turn=turn_index,
                        text=event.text,
                        detected_language_code=detected_language_code,
                        stt_provider_ms=_number(event.latency_ms),
                        endpoint_to_final_ms=_number(endpoint_to_final_ms),
                    )
                    emit("llm.started", turn=turn_index)
                    llm_started = time.perf_counter()
                    first_answer_at: float | None = None

                    def on_answer_chunk(chunk: str) -> None:
                        nonlocal first_answer_at
                        if first_answer_at is None:
                            first_answer_at = time.perf_counter()
                            emit_from_thread(
                                "llm.first_answer",
                                turn=turn_index,
                                transcript_to_first_answer_ms=round(
                                    (first_answer_at - committed_at) * 1000, 1
                                ),
                            )
                        emit_from_thread("llm.delta", turn=turn_index, text=chunk)

                    try:
                        result = await bridge.handle_async(event, on_chunk=on_answer_chunk)
                    except Exception as error:
                        emit("llm.error", turn=turn_index, error=type(error).__name__, message=str(error)[:500])
                        continue
                    completed_at = time.perf_counter()
                    if result is None:
                        emit("transcript.duplicate", turn=turn_index)
                        continue
                    answer = str(result.get("spoken") or "")
                    if first_answer_at is None:
                        first_answer_at = completed_at
                        if answer:
                            emit("llm.first_answer", turn=turn_index, transcript_to_first_answer_ms=round((first_answer_at - committed_at) * 1000, 1))
                            emit("llm.delta", turn=turn_index, text=answer)
                    graph_metrics = result.get("metrics", {})
                    transcript_to_first_ms = (
                        (first_answer_at - committed_at) * 1000 if first_answer_at is not None else None
                    )
                    speech_end_to_first_ms = (
                        stt.min_silence_ms + (endpoint_to_final_ms or 0.0) + transcript_to_first_ms
                        if transcript_to_first_ms is not None else None
                    )
                    metrics = {
                        "turn": turn_index,
                        "stt_provider_ms": _number(event.latency_ms),
                        "endpoint_to_final_ms": _number(endpoint_to_final_ms),
                        "routing_prompt_ms": _number(graph_metrics.get("intent_ms")),
                        "model_ttft_ms": _number(graph_metrics.get("llm_ttft_ms")),
                        "transcript_to_first_answer_ms": _number(transcript_to_first_ms),
                        "model_stream_ms": _number(graph_metrics.get("llm_stream_ms")),
                        "llm_pipeline_total_ms": round((completed_at - llm_started) * 1000, 1),
                        "estimated_speech_end_to_first_answer_ms": _number(speech_end_to_first_ms),
                    }
                    measurements.append(metrics)
                    emit(
                        "llm.completed",
                        turn=turn_index,
                        answer=answer,
                        intent=result.get("intent"),
                        topic=result.get("topic"),
                        metrics=metrics,
                    )

            incoming = asyncio.create_task(receive_audio())
            events = asyncio.create_task(process_stt_events())
            done, _ = await asyncio.wait({incoming, events}, return_when=asyncio.FIRST_COMPLETED)
            if incoming in done and incoming.result() == "stop":
                # Allow provider VAD to close the final utterance before the
                # diagnostic connection is torn down.
                await asyncio.sleep(max(1.0, stt.min_silence_ms / 1000 + 0.7))
            for task in (incoming, events):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

        emit(
            "session.completed",
            turns=len(measurements),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 1),
            measurements=measurements,
        )
        # Do not close the socket until the dedicated sender confirms that the
        # terminal event was written, not merely removed from the queue.
        await outgoing.join()
    except WebSocketDisconnect:
        return
    except Exception as error:
        emit("session.error", error=type(error).__name__, message=str(error)[:500])
        await asyncio.sleep(0.05)
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await sender
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()
