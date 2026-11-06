#!/usr/bin/env python3
"""Smoke-test the deployed VoiceLink WSS endpoint without placing a call.

Usage keeps the token out of shell history when VOICELINK_WS_URL is exported.
The test sends no caller audio, so it does not invoke the LLM or produce TTS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("VOICELINK_WS_URL"), help="Full WSS URL including ?token=...")
    args = parser.parse_args()
    if not args.url:
        parser.error("Set VOICELINK_WS_URL or pass --url")
    try:
        import websockets
    except ImportError as error:
        raise SystemExit("Install support with: uv sync --extra gnani-stt") from error

    async with websockets.connect(args.url, open_timeout=15) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({
            "event": "start",
            "sequence_number": 0,
            "stream_sid": "local-smoke-stream",
            "start": {
                "stream_sid": "local-smoke-stream",
                "call_sid": "local-smoke-call",
                "account_sid": "local",
                "from": "+910000000000",
                "to": "+910000000001",
                "custom_parameters": {"customer_id": "local-wss-smoke"},
                "media_format": {"encoding": "audio/alaw", "sample_rate": "8000"},
            },
        }))
        await asyncio.sleep(1)
        await ws.send(json.dumps({"event": "stop", "stop": {"callSid": "local-smoke-call"}}))
    print("VoiceLink WSS smoke test passed: authenticated upgrade, start, and stop accepted.")


if __name__ == "__main__":
    asyncio.run(main())
