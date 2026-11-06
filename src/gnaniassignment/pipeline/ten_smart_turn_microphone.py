"""Test TEN VAD plus Pipecat Smart Turn with a local microphone.

This probe is deliberately isolated from STT, LLM and TTS. It captures mono
16 kHz PCM, prints each short-frame acoustic transition, then asks the
semantic model whether a pause is a complete or incomplete user turn.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import time

from gnaniassignment.voice.ten_smart_turn import TenPipecatTurnDetector


async def run(device: str, seconds: float, threshold: float, hop_samples: int, min_speech_ms: int, min_silence_ms: int) -> None:
    detector = TenPipecatTurnDetector(
        threshold=threshold,
        hop_samples=hop_samples,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
    )
    command = [
        "arecord", "-D", device, "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw",
    ]
    print(f"Listening on {device!r} for {seconds:g}s. Try a mid-sentence pause, then finish a thought.", flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            assert process.stdout is not None
            pcm = process.stdout.read(512)  # 16 ms at 16 kHz.
            if not pcm:
                if process.poll() is not None:
                    assert process.stderr is not None
                    detail = process.stderr.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"arecord exited with status {process.returncode}: {detail}")
                continue
            for boundary in detector.feed_pcm_s16le(pcm):
                print(f"{time.strftime('%H:%M:%S')}  {boundary.kind}  acoustic={boundary.probability:.2f}", flush=True)
                if boundary.kind != "speech_ended":
                    continue
                decision = await detector.analyze_current_pause(detector.revision)
                if not decision.stale:
                    probability = "n/a" if decision.probability is None else f"{decision.probability:.2f}"
                    label = "complete" if decision.complete else "incomplete — keep listening"
                    print(f"{time.strftime('%H:%M:%S')}  smart_turn={label}  semantic={probability}", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        await detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="default", help="ALSA capture device, e.g. plughw:1,0")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hop-samples", type=int, choices=(160, 256), default=256)
    parser.add_argument("--min-speech-ms", type=int, default=160)
    parser.add_argument("--min-silence-ms", type=int, default=320)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    asyncio.run(run(args.device, args.seconds, args.threshold, args.hop_samples, args.min_speech_ms, args.min_silence_ms))


if __name__ == "__main__":
    main()
