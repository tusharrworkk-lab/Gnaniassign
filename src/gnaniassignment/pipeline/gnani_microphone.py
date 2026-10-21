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


async def pcm_frames(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Coalesce arbitrary ALSA pipe reads into Gnani's fixed 1,024-byte frames."""
    buffered = bytearray()
    while chunk := await reader.read(4096):
        buffered.extend(chunk)
        while len(buffered) >= PCM_FRAME_BYTES:
            yield bytes(buffered[:PCM_FRAME_BYTES])
            del buffered[:PCM_FRAME_BYTES]


async def run(seconds: int | None, device: str, with_llm: bool = False, with_tts: bool = False, playback_device: str | None = None) -> None:
    load_environment()
    stt = GnaniRealtimeStt()
    bridge = GnaniTurnBridge(create_session()) if with_llm else None
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
        async def receive() -> None:
            async for event in stt.events():
                if isinstance(event, GnaniProcessing):
                    print("stt: processing", flush=True)
                    write_voice_metric({"event": "stt_processing"})
                elif isinstance(event, GnaniTranscript):
                    turn_started = time.perf_counter()
                    transcripts.append(event)
                    print(f"transcript: {event.text}", flush=True)
                    print(f"stt_latency_ms: {event.latency_ms}", flush=True)
                    write_voice_metric({
                        "event": "stt_final",
                        "stt_provider_latency_ms": event.latency_ms,
                        "stt_audio_duration_ms": event.audio_duration_ms,
                    })
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
                            print(chunk, end="", flush=True)

                        result = await bridge.handle_async(event, on_chunk=show_answer)
                        if result is not None:
                            llm_completed_at = time.perf_counter()
                            print()
                            print(f"intent: {result['intent']}", flush=True)
                            record: dict[str, object] = {
                                "event": "voice_turn",
                                "intent": result["intent"],
                                "stt_provider_latency_ms": event.latency_ms,
                                "stt_audio_duration_ms": event.audio_duration_ms,
                                "pipeline_e2e_to_llm_complete_ms": (llm_completed_at - turn_started) * 1000,
                                **result.get("metrics", {}),
                            }
                            if first_answer_chunk_at is not None:
                                record["pipeline_e2e_to_llm_ttft_ms"] = (first_answer_chunk_at - turn_started) * 1000
                            if with_tts and result["spoken"]:
                                print("tts: speaking", flush=True)
                                tts_started = time.perf_counter()

                                def on_first_audio(metrics: dict[str, float | int | None]) -> None:
                                    live = {"event": "tts_first_audio", **metrics}
                                    if isinstance(metrics["tts_ttfa_ms"], (int, float)):
                                        live["pipeline_e2e_to_tts_first_audio_ms"] = (tts_started - turn_started) * 1000 + metrics["tts_ttfa_ms"]
                                    write_voice_metric(live)

                                tts_metrics = await speak(result["spoken"], playback_device, on_first_audio=on_first_audio)
                                record.update(tts_metrics)
                                if isinstance(tts_metrics["tts_ttfa_ms"], (int, float)):
                                    record["pipeline_e2e_to_tts_first_audio_ms"] = (tts_started - turn_started) * 1000 + tts_metrics["tts_ttfa_ms"]
                                if isinstance(tts_metrics["playback_first_write_ms"], (int, float)):
                                    record["pipeline_e2e_to_playback_first_write_ms"] = (tts_started - turn_started) * 1000 + tts_metrics["playback_first_write_ms"]
                                if isinstance(tts_metrics["tts_synthesis_ms"], (int, float)):
                                    record["pipeline_e2e_to_tts_stream_complete_ms"] = (tts_started - turn_started) * 1000 + tts_metrics["tts_synthesis_ms"]
                                # This includes the remaining local speaker buffer; it is
                                # not the conversational latency target.
                                record["pipeline_e2e_to_playback_complete_ms"] = (time.perf_counter() - turn_started) * 1000
                            path = write_voice_metric(record)
                            print(f"metrics_log: {path}", flush=True)
                elif isinstance(event, GnaniError):
                    print(f"stt_error: {event.message}", flush=True)

        receiver = asyncio.create_task(receive())
        try:
            await stt.send_frames(pcm_frames(process.stdout), realtime_pace=False)
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
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.with_tts and not args.with_llm:
        parser.error("--with-tts requires --with-llm")
    try:
        asyncio.run(run(args.seconds, args.device, with_llm=args.with_llm, with_tts=args.with_tts, playback_device=args.playback_device))
    except KeyboardInterrupt:
        print("call ended")


if __name__ == "__main__":
    main()
