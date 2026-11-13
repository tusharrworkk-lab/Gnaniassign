"""Generate a reusable raw 8 kHz A-law clear-throat cue from Gnani TTS."""

from __future__ import annotations

import argparse
import asyncio
import audioop
from pathlib import Path

from gnaniassignment.tts.gnani import GnaniRealtimeTts


async def build(destination: Path) -> None:
    output = bytearray()
    state = None
    async with GnaniRealtimeTts() as tts:
        async for chunk in tts.synthesize("", add_clear_throat=True):
            pcm, state = audioop.ratecv(chunk.audio, 2, 1, tts.sample_rate, 8000, state)
            output.extend(audioop.lin2alaw(pcm, 2))
    if not output:
        raise RuntimeError("Gnani returned no audio for the cue")
    # VoiceLink expects exact 160-byte frames. A-law D5 is digital silence.
    output.extend(b"\xd5" * (-len(output) % 160))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(output)
    print(f"Saved {len(output) // 160} VoiceLink frames to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(build(args.output))


if __name__ == "__main__":
    main()
