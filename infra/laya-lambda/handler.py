"""Private Lambda handler that keeps one CPU laya.cpp server warm per sandbox."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


_server: subprocess.Popen[bytes] | None = None
_lock = threading.Lock()


def _url(path: str) -> str:
    return f"http://127.0.0.1:{os.environ.get('LAYA_PORT', '8080')}{path}"


def _ensure_server() -> None:
    global _server
    with _lock:
        if _server is not None and _server.poll() is None:
            return
        _server = subprocess.Popen(
            [
                os.environ.get("LAYA_BINARY", "/opt/laya/laya-cli"),
                "--server", "--host", "127.0.0.1", "--port", os.environ.get("LAYA_PORT", "8080"),
                "--model", os.environ.get("LAYA_MODEL_DIR", "/opt/laya/models/laya"),
                "--variant", os.environ.get("LAYA_VARIANT", "multilingual"), "--cpu",
                "--batch-wait-ms", "0", "--max-pending-requests", "8",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if _server.poll() is not None:
                error = _server.stderr.read().decode("utf-8", errors="replace") if _server.stderr else ""
                raise RuntimeError(f"laya.cpp exited during initialization: {error}")
            try:
                with urlopen(_url("/health"), timeout=0.25) as response:
                    if response.status == 200:
                        return
            except URLError:
                time.sleep(0.05)
        raise TimeoutError("laya.cpp did not become ready within 25 seconds")


def _response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": status_code, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Accept a direct Lambda Invoke payload in laya.cpp's JEV request shape."""
    if not isinstance(event, dict):
        return _response(400, {"error": "Request body must be a JSON object"})
    if not isinstance(event.get("questions"), dict) or "state" not in event:
        return _response(400, {"error": "Request requires state and questions"})
    if context is not None and context.get_remaining_time_in_millis() < 1_000:
        return _response(503, {"error": "Insufficient Lambda time remaining"})
    try:
        _ensure_server()
        request = Request(
            _url("/v1/systemone"), data=json.dumps(event).encode("utf-8"),
            headers={"content-type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
        return _response(200, body)
    except Exception as error:
        return _response(503, {"error": f"Laya decision service unavailable: {error}"})
