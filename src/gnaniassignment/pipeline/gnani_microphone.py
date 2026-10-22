"""Capture Linux microphone PCM and stream it to Gnani without writing audio files."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import AsyncIterator

from gnaniassignment.app.config import load_environment
from gnaniassignment.app.container import create_session
from gnaniassignment.stt.gnani import GnaniError, GnaniProcessing, GnaniRealtimeStt, GnaniTranscript, GnaniTurnBridge, PCM_FRAME_BYTES
from gnaniassignment.pipeline.gnani_tts_speak import speak
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
) -> None:
    load_environment()
    stt = GnaniRealtimeStt()
    bridge = GnaniTurnBridge(create_session()) if with_llm else None
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
                            print(chunk, end="", flush=True)

                        result = await bridge.handle_async(event, on_chunk=show_answer)
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
                                    tts_metrics = await speak(result["spoken"], playback_device, on_first_audio=on_first_audio)
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
    print(f"final_segment_count: {len(transcripts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send live Linux microphone PCM to Gnani STT")
    parser.add_argument("--seconds", type=int, help="Optional capture duration; omit to keep the call connection open")
    parser.add_argument("--device", default="plughw:1,0", help="ALSA capture device")
    parser.add_argument("--with-llm", action="store_true", help="Route final STT segments through the intent and LLM pipeline")
    parser.add_argument("--with-tts", action="store_true", help="Speak each LLM reply through Gnani Timbre realtime TTS (requires --with-llm)")
    parser.add_argument("--playback-device", help="Optional ALSA playback device for --with-tts")
    parser.add_argument("--allow-barge-in", action="store_true", help="Keep forwarding microphone audio while TTS plays; disable only when using real echo cancellation or separate call tracks")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.with_tts and not args.with_llm:
        parser.error("--with-tts requires --with-llm")
    try:
        asyncio.run(run(args.seconds, args.device, with_llm=args.with_llm, with_tts=args.with_tts, playback_device=args.playback_device, suppress_stt_during_tts=not args.allow_barge_in))
    except KeyboardInterrupt:
        print("call ended")


if __name__ == "__main__":
    main()
