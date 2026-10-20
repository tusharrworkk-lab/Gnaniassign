"""Verify Gnani's negotiated STT configuration without sending audio."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from gnaniassignment.app.config import load_environment
from gnaniassignment.stt.gnani import GnaniRealtimeStt


async def probe() -> list[str]:
    """Open and validate the Gnani handshake; no PCM data is transmitted."""
    load_environment()
    stt = GnaniRealtimeStt()
    lines = [
        f"requested_language: {stt.language_code}",
        f"requested_sample_rate: {stt.sample_rate}",
        f"expected_frame_bytes: 1024",
        f"expected_frame_cadence_ms: {stt.frame_interval_seconds * 1000:.0f}",
        f"vad_min_silence_ms: {stt.min_silence_ms}",
    ]
    try:
        async with stt:
            lines.append(f"acknowledged_sample_rate: {stt.active_sample_rate}")
            lines.append("result: accepted")
    except RuntimeError as error:
        lines.append(f"result: rejected - {error}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the Gnani STT handshake configuration")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("artifacts/gnani-stt-setup.log"),
        help="Local append-only diagnostic log path",
    )
    args = parser.parse_args()
    lines = [f"[{datetime.now(timezone.utc).isoformat()}]"] + asyncio.run(probe()) + [""]
    output = "\n".join(lines)
    print(output)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as log_file:
        log_file.write(output)
        log_file.write("\n")
    print(f"log_file: {args.log_file}")


if __name__ == "__main__":
    main()
