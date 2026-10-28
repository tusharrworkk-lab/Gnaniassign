"""Capture Linux microphone PCM and stream it to Gnani without writing audio files."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections.abc import AsyncIterator

from gnaniassignment.app.config import load_environment
from gnaniassignment.app.container import create_session
from gnaniassignment.stt.gnani import GnaniError, GnaniProcessing, GnaniRealtimeStt, GnaniTranscript, GnaniTurnBridge, PCM_FRAME_BYTES
from gnaniassignment.pipeline.gnani_tts_speak import speak
from gnaniassignment.tts.gnani import should_add_clear_throat
from gnaniassignment.pipeline.voice_metrics import write_voice_metric


def _ms(value: object) -> str:
    """Render an optional latency value for live terminal diagnostics."""
    return f"{float(value):.0f} ms" if isinstance(value, (int, float)) else "n/a"


async def pcm_frames(reader: asyncio.StreamReader, input_enabled: asyncio.Event | None = None) -> AsyncIterator[bytes]:
    """Coalesce arbitrary ALSA pipe reads into Gnani's fixed 1,024-byte frames."""
    buffered = bytearray()
    while chunk := await reader.read(4096):
        # Desktop echo suppression: keep draining ALSA while TTS is speaking,
        # but never forward those microphone samples into the STT connection.
        if input_enabled is not None and not input_enabled.is_set():
            buffered.clear()
            continue
        buffered.extend(chunk)
        while len(buffered) >= PCM_FRAME_BYTES:
            if input_enabled is not None and not input_enabled.is_set():
                buffered.clear()
                break
            yield bytes(buffered[:PCM_FRAME_BYTES])
            del buffered[:PCM_FRAME_BYTES]


async def run(
    seconds: int | None,
    device: str,
    with_llm: bool = False,
    with_tts: bool = False,
    playback_device: str | None = None,
    suppress_stt_during_tts: bool = True,
    customer_id: str | None = None,
) -> None:
    load_environment()
    stt = GnaniRealtimeStt()
    bridge = GnaniTurnBridge(create_session(customer_id=customer_id)) if with_llm else None
    input_enabled = asyncio.Event()
    input_enabled.set()
    arecord_command = [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(stt.sample_rate),
        "-t",
        "raw",
    ]
    if seconds is not None:
        arecord_command.extend(["-d", str(seconds)])
    process = await asyncio.create_subprocess_exec(*arecord_command, stdout=asyncio.subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Could not read microphone PCM output")

    transcripts: list[GnaniTranscript] = []
    async with stt:
        processing_received_at: float | None = None

        async def receive() -> None:
            nonlocal processing_received_at
            async for event in stt.events():
                if isinstance(event, GnaniProcessing):
                    processing_received_at = time.perf_counter()
                    print("stt: processing", flush=True)
                    write_voice_metric({"event": "stt_processing"})
                elif isinstance(event, GnaniTranscript):
                    final_received_at = time.perf_counter()
                    turn_started = final_received_at
                    transcripts.append(event)
                    if bridge is not None and hasattr(bridge.session, "record_stt_segment"):
                        bridge.session.record_stt_segment(event.text, event.segment_id, event.latency_ms)
                    print(f"transcript: {event.text}", flush=True)
                    print(f"metrics | STT final: {_ms(event.latency_ms)} | end-of-turn target: {stt.min_silence_ms} ms", flush=True)
                    stt_metrics: dict[str, float | int | None] = {
                        "event": "stt_final",
                        "end_of_turn_target_ms": stt.min_silence_ms,
                        "transcription_provider_latency_ms": event.latency_ms,
                    }
                    if processing_received_at is not None:
                        stt_metrics["transcription_post_vad_to_final_ms"] = (final_received_at - processing_received_at) * 1000
                        print(
                            f"metrics | VAD close → final transcript: {_ms(stt_metrics['transcription_post_vad_to_final_ms'])}",
                            flush=True,
                        )
                    write_voice_metric(stt_metrics)
                    processing_received_at = None
                    if bridge is not None and event.text.strip():
                        print("pipeline: routing and generating", flush=True)
                        first_answer_chunk_at: float | None = None
                        first_answer_chunk = asyncio.Event()
                        reply_playback_gate = asyncio.Event()
                        loop = asyncio.get_running_loop()

                        def show_answer(chunk: str) -> None:
                            nonlocal first_answer_chunk_at
                            if first_answer_chunk_at is None:
                                first_answer_chunk_at = time.perf_counter()
                                write_voice_metric({
                                    "event": "llm_first_token",
                                    "pipeline_e2e_to_llm_ttft_ms": (first_answer_chunk_at - turn_started) * 1000,
                                })
                                print(
                                    f"metrics | LLM TTFT: {_ms((first_answer_chunk_at - turn_started) * 1000)}",
                                    flush=True,
                                )
                                loop.call_soon_threadsafe(first_answer_chunk.set)
                            print(chunk, end="", flush=True)

                        generation = asyncio.create_task(bridge.handle_async(event, on_chunk=show_answer))

                        async def play_latency_giggle() -> bool:
                            """Fill long pre-token silence without involving the LLM."""
                            if not with_tts or os.getenv("GNANI_TTS_LATENCY_GIGGLE_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
                                reply_playback_gate.set()
                                return False
                            threshold_ms = float(os.getenv("GNANI_TTS_LATENCY_GIGGLE_MS", "450"))
                            try:
                                await asyncio.wait_for(first_answer_chunk.wait(), timeout=threshold_ms / 1000)
                                reply_playback_gate.set()
                                return False
                            except asyncio.TimeoutError:
                                if generation.done():
                                    reply_playback_gate.set()
                                    return False
                                print("tts: latency giggle", flush=True)
                                write_voice_metric({"event": "tts_latency_giggle", "threshold_ms": threshold_ms})
                                if suppress_stt_during_tts:
                                    input_enabled.clear()
                                try:
                                    await speak("<giggle>", playback_device, latency_giggle=True)
                                finally:
                                    input_enabled.set()
                                    reply_playback_gate.set()
                                return True

                        latency_giggle = asyncio.create_task(play_latency_giggle())
                        result = await generation
                        if result is not None:
                            print()
                            print(f"intent: {result['intent']}", flush=True)
                            record: dict[str, object] = {
                                key: value for key, value in stt_metrics.items() if key != "event"
                            }
                            record["event"] = "voice_turn"
                            if first_answer_chunk_at is not None:
                                record["llm_ttft_ms"] = result.get("metrics", {}).get("llm_ttft_ms")
                                record["e2e_final_transcript_to_llm_ttft_ms"] = (first_answer_chunk_at - turn_started) * 1000
                            if with_tts and result["spoken"]:
                                print("tts: speaking", flush=True)
                                tts_started = time.perf_counter()
                                llm_ttft_ms = (first_answer_chunk_at - turn_started) * 1000 if first_answer_chunk_at is not None else result.get("metrics", {}).get("llm_ttft_ms")

                                def on_first_audio(metrics: dict[str, float | int | None]) -> None:
                                    live = {"event": "tts_first_audio", **metrics}
                                    if isinstance(metrics["tts_ttfa_ms"], (int, float)):
                                        live["pipeline_e2e_to_tts_first_audio_ms"] = (tts_started - turn_started) * 1000 + metrics["tts_ttfa_ms"]
                                    write_voice_metric(live)
                                    print(
                                        "metrics | "
                                        f"TTS TTFB: {_ms(metrics['tts_ttfa_ms'])} | "
                                        f"final transcript → audible audio: {_ms(live.get('pipeline_e2e_to_tts_first_audio_ms'))}",
                                        flush=True,
                                    )

                                if suppress_stt_during_tts:
                                    input_enabled.clear()
                                    write_voice_metric({"event": "stt_suppressed_during_tts"})
                                try:
                                    # The request starts now, even if the latency cue is
                                    # still audible. `playback_gate` preserves cue → reply
                                    # ordering while hiding the answer TTS setup time.
                                    answer_tts = asyncio.create_task(speak(
                                        result["spoken"], playback_device, on_first_audio=on_first_audio,
                                        add_clear_throat=(
                                            os.getenv("GNANI_TTS_POST_RESPONSE_GIGGLE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
                                            and should_add_clear_throat(
                                                llm_ttft_ms,
                                                float(os.getenv("GNANI_TTS_POST_RESPONSE_GIGGLE_MS", "500")),
                                            )
                                        ),
                                        playback_gate=reply_playback_gate,
                                    ))
                                    tts_metrics = await answer_tts
                                finally:
                                    input_enabled.set()
                                if isinstance(tts_metrics["tts_ttfa_ms"], (int, float)):
                                    record["tts_ttfb_ms"] = tts_metrics["tts_ttfa_ms"]
                                if isinstance(tts_metrics["playback_first_write_ms"], (int, float)):
                                    e2e = (tts_started - turn_started) * 1000 + tts_metrics["playback_first_write_ms"]
                                    record["e2e_final_transcript_to_playback_ms"] = e2e
                                    if isinstance(record.get("transcription_post_vad_to_final_ms"), (int, float)):
                                        record["estimated_e2e_speech_end_to_playback_ms"] = (
                                            stt.min_silence_ms + record["transcription_post_vad_to_final_ms"] + e2e
                                        )
                            path = write_voice_metric(record)
                            print(
                                "metrics | turn summary | "
                                f"LLM TTFT: {_ms(record.get('llm_ttft_ms'))} | "
                                f"TTS TTFB: {_ms(record.get('tts_ttfb_ms'))} | "
                                f"speech-end → audible audio: {_ms(record.get('estimated_e2e_speech_end_to_playback_ms'))}",
                                flush=True,
                            )
                            print(f"metrics_log: {path}", flush=True)
                elif isinstance(event, GnaniError):
                    print(f"stt_error: {event.message}", flush=True)

        receiver = asyncio.create_task(receive())
        try:
            await stt.send_frames(pcm_frames(process.stdout, input_enabled), realtime_pace=False)
            await process.wait()
            if process.returncode:
                raise RuntimeError(f"arecord exited with status {process.returncode}")
            # Allow VAD closure and the final transcript to arrive after capture ends.
            await asyncio.sleep(max(1.0, stt.min_silence_ms / 1000 + 0.7))
        finally:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
            if process.returncode is None:
                process.terminate()
                await process.wait()
            # This is deliberately inside finally: Ctrl+C must still persist
            # the STT/LLM transcript and run the end-of-call judge.
            if bridge is not None and hasattr(bridge.session, "finalize_call"):
                judgment = await asyncio.to_thread(bridge.session.finalize_call)
                if judgment is not None:
                    print(f"call_judgment: {'failed' if 'judge_error' in judgment else 'completed'}", flush=True)
    print(f"final_segment_count: {len(transcripts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send live Linux microphone PCM to Gnani STT")
    parser.add_argument("--seconds", type=int, help="Optional capture duration; omit to keep the call connection open")
    parser.add_argument("--device", default="plughw:1,0", help="ALSA capture device")
    parser.add_argument("--with-llm", action="store_true", help="Route final STT segments through the intent and LLM pipeline")
    parser.add_argument("--with-tts", action="store_true", help="Speak each LLM reply through Gnani Timbre realtime TTS (requires --with-llm)")
    parser.add_argument("--playback-device", help="Optional ALSA playback device for --with-tts")
    parser.add_argument("--customer-id", help="Authenticated caller/customer ID; loads that account's order window before the call")
    parser.add_argument("--allow-barge-in", action="store_true", help="Keep forwarding microphone audio while TTS plays; disable only when using real echo cancellation or separate call tracks")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.with_tts and not args.with_llm:
        parser.error("--with-tts requires --with-llm")
    try:
        asyncio.run(run(args.seconds, args.device, with_llm=args.with_llm, with_tts=args.with_tts, playback_device=args.playback_device, suppress_stt_during_tts=not args.allow_barge_in, customer_id=args.customer_id))
    except KeyboardInterrupt:
        print("call ended")


if __name__ == "__main__":
    main()
