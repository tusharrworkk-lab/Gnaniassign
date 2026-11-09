"""Deterministic scoring for privacy-preserving voice-agent evaluations."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def percentile(values: Iterable[float], percent: int) -> float | None:
    """Nearest-rank percentile, deliberately dependency-free and reproducible."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, -(-len(ordered) * percent // 100) - 1))
    return round(ordered[index], 1)


def _words(text: str) -> list[str]:
    """Normalize enough for Roman-script Hinglish WER without hiding errors."""
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> dict[str, float | int | None]:
    """Return word-level Levenshtein WER for a labelled STT sample."""
    expected, actual = _words(reference), _words(hypothesis)
    if not expected:
        return {"reference_words": 0, "errors": 0, "wer": None}
    prior = list(range(len(actual) + 1))
    for index, expected_word in enumerate(expected, start=1):
        current = [index]
        for actual_index, actual_word in enumerate(actual, start=1):
            current.append(min(
                prior[actual_index] + 1,
                current[actual_index - 1] + 1,
                prior[actual_index - 1] + (expected_word != actual_word),
            ))
        prior = current
    errors = prior[-1]
    return {"reference_words": len(expected), "errors": errors, "wer": round(errors / len(expected), 4)}


def summarize_turns(records: Iterable[dict[str, Any]], silent_after_ms: int = 2500) -> dict[str, Any]:
    """Summarize metadata-only live or replayed voice-turn records.

    ``first_outbound_audio_ms`` is a server-to-telephony hand-off proxy. A
    telephone provider does not report actual handset playback, so this module
    never labels it as caller-heard audio.
    """
    rows = [record.get("metrics", record) for record in records]
    completed = [row for row in rows if row.get("evaluation_finalized")]
    first_audio_latencies = [
        row["first_outbound_audio_latency_ms"]
        for row in completed
        if isinstance(row.get("first_outbound_audio_latency_ms"), (int, float))
    ]
    llm_ttfts = [
        row["llm_ttft_latency_ms"]
        for row in completed
        if isinstance(row.get("llm_ttft_latency_ms"), (int, float))
    ]
    silent = [
        row for row in completed
        if row.get("silent_turn") is True
        or not isinstance(row.get("first_outbound_audio_latency_ms"), (int, float))
        or row["first_outbound_audio_latency_ms"] > silent_after_ms
    ]
    annotated = [row for row in completed if isinstance(row.get("reference_turn_end_ms"), (int, float))]
    premature = [
        row for row in annotated
        if isinstance(row.get("first_outbound_audio_at_ms"), (int, float))
        and row["first_outbound_audio_at_ms"] < row["reference_turn_end_ms"]
    ]
    barge_attempts = [row for row in completed if isinstance(row.get("barge_in_detected_at_ms"), (int, float))]
    barge_stop_latencies = [
        row["barge_in_stop_latency_ms"]
        for row in barge_attempts
        if isinstance(row.get("barge_in_stop_latency_ms"), (int, float))
    ]

    def distribution(values: list[float]) -> dict[str, float | int | None]:
        return {"count": len(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99)}

    return {
        "definition": {
            "first_audible_reply": "first outbound media frame handed to VoiceLink after a committed user turn; handset playback is not observable",
            "silent_turn": f"no first outbound frame within {silent_after_ms} ms after a committed user turn",
            "premature_turn": "requires an independently labelled reference_turn_end_ms; not inferred from VAD alone",
            "barge_in_stop": "time from detected caller speech while agent output is active to the pipeline's clear/stop hand-off",
        },
        "turns": {"observed": len(rows), "completed": len(completed)},
        "first_audible_reply_proxy": distribution(first_audio_latencies),
        "llm_ttft": distribution(llm_ttfts),
        "silent_turn_rate": round(len(silent) / len(completed), 4) if completed else None,
        "silent_turns": len(silent),
        "premature_turn_rate": round(len(premature) / len(annotated), 4) if annotated else None,
        "premature_turns": len(premature),
        "premature_turns_annotated": len(annotated),
        "barge_in_attempts": len(barge_attempts),
        "barge_in_stop_latency": distribution(barge_stop_latencies),
    }


def summarize_wer(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate labelled STT samples without depending on a provider SDK."""
    samples = [row for row in rows if isinstance(row.get("reference_text"), str) and isinstance(row.get("hypothesis_text"), str)]
    scores = [word_error_rate(row["reference_text"], row["hypothesis_text"]) for row in samples]
    words = sum(int(score["reference_words"]) for score in scores)
    errors = sum(int(score["errors"]) for score in scores)
    return {"samples": len(scores), "reference_words": words, "errors": errors, "wer": round(errors / words, 4) if words else None}
