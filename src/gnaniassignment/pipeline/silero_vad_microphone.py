"""Listen to a local microphone and print Silero VAD boundaries.

This is a local diagnostic only. It never connects to VoiceLink, Gnani STT,
TTS, or an LLM. Capture settings intentionally mirror VoiceLink media: mono,
16-bit PCM at 8 kHz.
"""

from __future__ import annotations

import argparse
import subprocess
import time

from gnaniassignment.voice.silero_vad import SileroVad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="default", help="ALSA capture device, e.g. plughw:1,0")
    parser.add_argument("--seconds", type=float, default=30, help="How long to observe audio")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-silence-ms", type=int, default=350)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    vad = SileroVad(8000, threshold=args.threshold, min_silence_ms=args.min_silence_ms)
    command = [
        "arecord", "-D", args.device, "-q", "-f", "S16_LE", "-r", "8000", "-c", "1", "-t", "raw",
    ]
    print(f"Listening on {args.device!r} for {args.seconds:g}s. Speak, pause, then speak again.", flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            assert process.stdout is not None
            pcm = process.stdout.read(512)  # One 32 ms Silero frame at 8 kHz.
            if not pcm:
                if process.poll() is not None:
                    raise RuntimeError(f"arecord exited with status {process.returncode}")
                continue
            for boundary in vad.feed_pcm_s16le(pcm):
                print(f"{time.strftime('%H:%M:%S')}  {boundary.kind}", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)


if __name__ == "__main__":
    main()
