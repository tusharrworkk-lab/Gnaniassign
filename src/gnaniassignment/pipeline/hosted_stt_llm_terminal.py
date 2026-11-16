"""Stream a local microphone through the hosted STT -> LLM diagnostic API."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from gnaniassignment.app.config import load_environment


def _write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False) + "\n")


def _latency(value: object) -> str:
    return f"{float(value):.1f} ms" if isinstance(value, (int, float)) else "n/a"


def _show_metrics(metrics: dict[str, Any]) -> None:
    print("\nlatency (production server)", flush=True)
    for label, key in (
        ("STT provider", "stt_provider_ms"),
        ("endpoint -> transcript", "endpoint_to_final_ms"),
        ("routing + prompt", "routing_prompt_ms"),
        ("model TTFT", "model_ttft_ms"),
        ("answer decode TTFT", "answer_decode_ttft_ms"),
        ("transcript -> first answer", "transcript_to_first_answer_ms"),
        ("model stream", "model_stream_ms"),
        ("complete LLM pipeline", "llm_pipeline_total_ms"),
        ("speech end -> first answer", "estimated_speech_end_to_first_answer_ms"),
    ):
        print(f"  {label:<29} {_latency(metrics.get(key))}", flush=True)


async def run(
    *,
    url: str,
    api_key: str,
    input_device: str,
    customer_id: str,
    seconds: int | None,
    persist: bool,
    events_file: Path,
) -> None:
    try:
        import websockets
    except ImportError as error:
        raise RuntimeError("Install WebSocket support with: uv sync --extra gnani-stt") from error

    header_name = (
        "additional_headers"
        if "additional_headers" in inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )
    print(f"Connecting to {url}", flush=True)
    async with websockets.connect(
        url,
        **{header_name: {"X-API-Key": api_key}},
        max_size=None,
        open_timeout=15,
    ) as websocket:
        await websocket.send(json.dumps({
            "type": "start",
            "customer_id": customer_id,
            "persist": persist,
        }))

        configuration: dict[str, Any] | None = None
        while configuration is None:
            event = json.loads(await websocket.recv())
            _write_event(events_file, event)
            if event.get("type") == "session.error":
                raise RuntimeError(event.get("message", "Hosted diagnostic failed"))
            if event.get("type") == "connected":
                configuration = event

        stt = configuration["stt"]
        sample_rate = int(stt["sample_rate"])
        frame_bytes = int(stt["frame_bytes"])
        print(
            f"Connected: {stt.get('model')} / {stt.get('language_code')} / "
            f"{sample_rate} Hz / {stt.get('min_silence_ms')} ms endpoint",
            flush=True,
        )
        print(
            f"Production composition: {_latency(configuration.get('composition_ms'))}; "
            f"persistent history: {configuration.get('persisted')}",
            flush=True,
        )
        print("Speak normally. Agent text will appear here; no TTS is generated.\n", flush=True)

        capture = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", input_device, "-f", "S16_LE", "-c", "1",
            "-r", str(sample_rate), "-t", "raw",
            *(["-d", str(seconds)] if seconds is not None else []),
            stdout=asyncio.subprocess.PIPE,
        )
        if capture.stdout is None:
            raise RuntimeError("Could not open microphone capture")

        async def send_audio() -> None:
            pending = bytearray()
            while chunk := await capture.stdout.read(4096):
                pending.extend(chunk)
                while len(pending) >= frame_bytes:
                    await websocket.send(bytes(pending[:frame_bytes]))
                    del pending[:frame_bytes]
            await capture.wait()
            if capture.returncode:
                raise RuntimeError(f"arecord exited with status {capture.returncode}")
            await websocket.send(json.dumps({"type": "stop"}))

        async def receive_events() -> None:
            active_turn: int | None = None
            async for raw in websocket:
                event = json.loads(raw)
                _write_event(events_file, event)
                event_type = event.get("type")
                if event_type == "stt.processing":
                    print("[STT] endpoint closed; transcribing...", flush=True)
                elif event_type == "transcript.partial":
                    print(f"\r[partial] {event.get('text', '')[:100]}", end="", flush=True)
                elif event_type == "transcript.final":
                    active_turn = event.get("turn")
                    language = event.get("detected_language_code") or "unknown"
                    print(f"\nTURN {active_turn}", flush=True)
                    print(f"you [{language}]: {event.get('text', '')}", flush=True)
                    print("agent: ", end="", flush=True)
                elif event_type == "llm.delta":
                    print(event.get("text", ""), end="", flush=True)
                elif event_type == "llm.completed":
                    print(flush=True)
                    print(f"intent: {event.get('intent')}", flush=True)
                    _show_metrics(event.get("metrics") or {})
                elif event_type in {"stt.error", "llm.error", "input.error", "session.error"}:
                    print(f"\n[{event_type}] {event.get('message', '')}", flush=True)
                elif event_type == "session.completed":
                    print(
                        f"\nProduction diagnostic complete: {event.get('turns')} turns, "
                        f"server session {_latency(event.get('elapsed_ms'))}",
                        flush=True,
                    )
                    return

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(receive_events())
        try:
            done, pending = await asyncio.wait(
                {sender, receiver}, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                task.result()
            if sender in done and receiver in pending:
                await receiver
        finally:
            for task in (sender, receiver):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            if capture.returncode is None:
                capture.terminate()
                with suppress(ProcessLookupError):
                    await capture.wait()


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(
        description="Use a local microphone against production STT and LLM without TTS",
    )
    parser.add_argument(
        "--url",
        default=os.getenv(
            "STT_LLM_DIAGNOSTIC_URL",
            "wss://voice.sidecut.me/v1/diagnostics/stt-llm",
        ),
    )
    parser.add_argument("--api-key", default=os.getenv("AGENT_API_KEY"))
    parser.add_argument("--input-device", default="default")
    parser.add_argument("--customer-id", default="diagnostic-local")
    parser.add_argument("--seconds", type=int, help="Capture duration; omit and press Ctrl+C")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Request durable history (server must separately allow this)",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=Path("artifacts/hosted-stt-llm-events.jsonl"),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set AGENT_API_KEY or pass --api-key")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        asyncio.run(run(
            url=args.url,
            api_key=args.api_key,
            input_device=args.input_device,
            customer_id=args.customer_id,
            seconds=args.seconds,
            persist=args.persist,
            events_file=args.events_file,
        ))
    except KeyboardInterrupt:
        print("\nStopped hosted STT -> LLM diagnostic.", flush=True)


if __name__ == "__main__":
    main()
