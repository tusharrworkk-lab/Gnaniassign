"""Structured, audio-free latency logging for voice turns."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_voice_metric(record: dict[str, Any]) -> Path:
    """Append one JSONL timing record without recording customer audio/text."""
    path = Path(os.getenv("VOICE_METRICS_LOG", "artifacts/voice-latency.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {"timestamp": datetime.now(UTC).isoformat(), **record}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(safe, separators=(",", ":"), allow_nan=False) + "\n")
    return path
