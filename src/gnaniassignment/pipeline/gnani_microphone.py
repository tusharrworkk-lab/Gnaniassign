"""Capture Linux microphone PCM and stream it to Gnani without writing audio files."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator

from gnaniassignment.app.config import load_environment
from gnaniassignment.stt.gnani import GnaniError, GnaniProcessing, GnaniRealtimeStt, GnaniTranscript, PCM_FRAME_BYTES


async def pcm_frames(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Coalesce arbitrary ALSA pipe reads into Gnani's fixed 1,024-byte frames."""
    buffered = bytearray()
    while chunk := await reader.read(4096):
        buffered.extend(chunk)
        while len(buffered) >= PCM_FRAME_BYTES:
            yield bytes(buffered[:PCM_FRAME_BYTES])
            del buffered[:PCM_FRAME_BYTES]


async def run(seconds: int | None, device: str) -> None:
    load_environment()
    stt = GnaniRealtimeStt()
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
                elif isinstance(event, GnaniTranscript):
                    transcripts.append(event)
                    print(f"transcript: {event.text}", flush=True)
                    print(f"stt_latency_ms: {event.latency_ms}", flush=True)
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
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        asyncio.run(run(args.seconds, args.device))
    except KeyboardInterrupt:
        print("call ended")


if __name__ == "__main__":
    main()
