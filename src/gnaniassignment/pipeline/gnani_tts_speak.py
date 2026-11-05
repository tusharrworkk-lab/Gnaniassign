"""Speak a short response through Gnani realtime TTS without saving audio."""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable
from typing import Any

from gnaniassignment.app.config import load_environment
from gnaniassignment.voice.providers import create_tts_provider


async def speak(
    text: str,
    device: str | None = None,
    on_first_audio: Callable[[dict[str, float | int | None]], None] | None = None,
    add_clear_throat: bool = False,
    latency_giggle: bool = False,
    playback_gate: asyncio.Event | None = None,
) -> dict[str, float | int | None]:
    """Stream Gnani raw PCM into ALSA ``aplay`` as it arrives.

    ``playback_gate`` lets synthesis run while another short cue is audible.
    PCM is buffered only until the gate opens, so the reply can start directly
    after that cue without paying a second TTS first-audio delay.
    """
    load_environment()
    async with create_tts_provider() as tts:
        started = time.perf_counter()
        first_playback_ms: float | None = None
        command = ["aplay", "-q"]
        if device:
            command.extend(["-D", device])
        command.extend(["-f", "S16_LE", "-c", "1", "-r", str(tts.sample_rate), "-t", "raw"])
        player = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE)
        if player.stdin is None:
            raise RuntimeError("Could not open ALSA playback input")
        try:
            buffered_audio: list[bytes] = []

            async def write_audio(audio: bytes) -> None:
                nonlocal first_playback_ms
                player.stdin.write(audio)
                await player.stdin.drain()
                if first_playback_ms is None:
                    first_playback_ms = (time.perf_counter() - started) * 1000
                    if on_first_audio is not None:
                        on_first_audio({
                            "tts_ttfa_ms": tts.last_first_audio_ms,
                            "playback_first_write_ms": first_playback_ms,
                        })

            async for chunk in tts.synthesize(text, add_clear_throat=add_clear_throat, latency_giggle=latency_giggle):
                if playback_gate is not None and not playback_gate.is_set():
                    buffered_audio.append(chunk.audio)
                    continue
                if buffered_audio:
                    for audio in buffered_audio:
                        await write_audio(audio)
                    buffered_audio.clear()
                await write_audio(chunk.audio)
            if buffered_audio:
                if playback_gate is not None:
                    await playback_gate.wait()
                for audio in buffered_audio:
                    await write_audio(audio)
        finally:
            player.stdin.close()
            await player.stdin.wait_closed()
            await player.wait()
        if player.returncode:
            raise RuntimeError(f"aplay exited with status {player.returncode}")
        return {
            "tts_ttfa_ms": tts.last_first_audio_ms,
            "tts_synthesis_ms": tts.last_synthesis_ms,
            "tts_audio_chunks": tts.last_audio_chunks,
            "tts_audio_bytes": tts.last_audio_bytes,
            "tts_audio_duration_ms": (tts.last_audio_bytes / (tts.sample_rate * 2)) * 1000,
            "playback_first_write_ms": first_playback_ms,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Speak text through Gnani Timbre realtime TTS")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--device", help="Optional ALSA playback device")
    args = parser.parse_args()
    asyncio.run(speak(args.text, args.device))


if __name__ == "__main__":
    main()
