"""Use a local microphone and speakers against the hosted VoiceLink WSS API.

This is a development simulator, not a telephony provider replacement. It
sends VoiceLink's documented ``connected``, ``start``, ``media`` and ``stop``
events, with A-law 8 kHz audio, so the cloud media adapter can be exercised
without placing a real phone call.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import os
from contextlib import suppress
from uuid import uuid4


SAMPLE_RATE = 8000
PCM_FRAME_BYTES = 320  # 20 ms of 16-bit, mono, 8 kHz PCM.


def _event(name: str, **payload: object) -> str:
    return json.dumps({"event": name, **payload}, separators=(",", ":"))


async def run(url: str, token: str, input_device: str, playback_device: str, customer_id: str, seconds: int | None) -> None:
    try:
        import websockets
        from websockets.exceptions import ConnectionClosed
    except ImportError as error:
        raise RuntimeError("Install WebSocket support with: uv sync --extra gnani-stt") from error

    stream_sid = f"local-stream-{uuid4()}"
    call_sid = f"local-call-{uuid4()}"
    separator = "&" if "?" in url else "?"
    connect_url = f"{url}{separator}token={token}"
    capture = await asyncio.create_subprocess_exec(
        "arecord", "-q", "-D", input_device, "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-t", "raw",
        *( ["-d", str(seconds)] if seconds is not None else [] ),
        stdout=asyncio.subprocess.PIPE,
    )
    playback = await asyncio.create_subprocess_exec(
        "aplay", "-q", "-D", playback_device, "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-t", "raw",
        stdin=asyncio.subprocess.PIPE,
    )
    if capture.stdout is None or playback.stdin is None:
        raise RuntimeError("Could not open local audio pipes")

    print(f"connecting to hosted VoiceLink adapter as {stream_sid}", flush=True)
    input_enabled = asyncio.Event()
    input_enabled.set()
    try:
        async with websockets.connect(connect_url, max_size=None) as websocket:
            await websocket.send(_event("connected"))
            await websocket.send(_event(
                "start",
                sequence_number=0,
                stream_sid=stream_sid,
                start={
                    "stream_sid": stream_sid,
                    "call_sid": call_sid,
                    "account_sid": "local-simulator",
                    "from": "+910000000000",
                    "to": "+910000000001",
                    "custom_parameters": {"customer_id": customer_id},
                    "media_format": {"encoding": "audio/alaw", "sample_rate": "8000"},
                },
            ))

            async def send_microphone() -> None:
                buffer = bytearray()
                sequence = 1
                while chunk := await capture.stdout.read(4096):
                    buffer.extend(chunk)
                    while len(buffer) >= PCM_FRAME_BYTES:
                        pcm = bytes(buffer[:PCM_FRAME_BYTES])
                        del buffer[:PCM_FRAME_BYTES]
                        # A laptop microphone otherwise hears its own speaker
                        # playback and creates a synthetic caller/agent loop.
                        if not input_enabled.is_set():
                            continue
                        alaw = audioop.lin2alaw(pcm, 2)
                        try:
                            await websocket.send(_event("media", sequence_number=sequence, media={"track": "inbound", "payload": base64.b64encode(alaw).decode("ascii")}))
                        except ConnectionClosed:
                            return
                        sequence += 1
                        # Match VoiceLink's real 20 ms media cadence rather
                        # than flooding the cloud STT socket.
                        await asyncio.sleep(0.02)

            async def play_agent_audio() -> None:
                async for raw in websocket:
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if message.get("event") == "mark":
                        print("agent response completed", flush=True)
                        # Keep suppressing capture briefly while the local
                        # speaker's final buffer drains.
                        await asyncio.sleep(0.5)
                        input_enabled.set()
                        continue
                    media = message.get("media")
                    payload = media.get("payload") if isinstance(media, dict) else None
                    if message.get("event") != "media" or not isinstance(payload, str):
                        continue
                    try:
                        pcm = audioop.alaw2lin(base64.b64decode(payload, validate=True), 2)
                    except (ValueError, TypeError):
                        continue
                    input_enabled.clear()
                    playback.stdin.write(pcm)
                    await playback.stdin.drain()

            sender = asyncio.create_task(send_microphone())
            receiver = asyncio.create_task(play_agent_audio())
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            # When a fixed microphone capture ends, keep the socket alive
            # briefly so STT VAD, the LLM, and returned TTS audio can finish.
            if sender in done and not sender.cancelled() and sender.exception() is None:
                await asyncio.sleep(4)
            for task in pending:
                if not task.done():
                    task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
            with suppress(ConnectionClosed):
                await websocket.send(_event("stop", stop={"callSid": call_sid}))
    finally:
        for process in (capture, playback):
            if process.returncode is None:
                process.terminate()
                with suppress(ProcessLookupError):
                    await process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send local microphone audio through the hosted VoiceLink WSS adapter")
    parser.add_argument("--url", default=os.getenv("VOICELINK_MEDIA_URL", "wss://voice.sidecut.me/v1/media/voicelink"))
    parser.add_argument("--token", default=os.getenv("VOICELINK_WS_TOKEN"), help="Hosted WebSocket token; defaults to VOICELINK_WS_TOKEN")
    parser.add_argument("--input-device", default="default", help="ALSA capture device; inspect with arecord -L")
    parser.add_argument("--playback-device", default="default", help="ALSA playback device; inspect with aplay -L")
    parser.add_argument("--customer-id", default="local-voicelink-test")
    parser.add_argument("--seconds", type=int, help="Optional capture duration; omit and press Ctrl+C to stop")
    args = parser.parse_args()
    if not args.token:
        parser.error("Set VOICELINK_WS_TOKEN or pass --token")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        asyncio.run(run(args.url.rstrip("/"), args.token, args.input_device, args.playback_device, args.customer_id, args.seconds))
    except KeyboardInterrupt:
        print("local VoiceLink simulation stopped", flush=True)


if __name__ == "__main__":
    main()
