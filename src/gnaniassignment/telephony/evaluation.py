"""Deterministic scoring for privacy-preserving voice-agent evaluations."""

from __future__ import annotations

import re
from collections import Counter
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


CURRENT_SCHEMA_VERSION = 3


def conversation_quality_flags(
    user_text: str,
    response_text: str,
    *,
    turn_index: int,
    previous_response: str = "",
) -> dict[str, bool]:
    """Return conservative, metadata-safe quality signals for one turn.

    These are deterministic screening signals, not semantic ground truth.
    They make likely regressions reviewable without storing prompt contents in
    the metric row. Human/LLM-judge evaluation can later label the same turns.
    """
    user = " ".join(_words(user_text))
    response = " ".join(_words(response_text))
    previous = " ".join(_words(previous_response))
    greeting = bool(re.match(r"^(namaste|hello|hi)\b", response))
    correction = bool(re.match(r"^(nahi|nahin|no|mera matlab|i mean)\b", user))
    privacy_request = any(term in user for term in (
        "location", "address", "password", "otp", "schedule", "calendar",
    ))
    privacy_refusal = any(term in response for term in (
        "share nahi", "nahi bata", "private", "permission", "cannot share",
    ))
    current_question = response_text.rstrip().endswith("?")
    previous_question = previous_response.rstrip().endswith("?")
    current_terms, previous_terms = set(_words(response_text)), set(_words(previous_response))
    overlap = (
        len(current_terms & previous_terms) / max(1, len(current_terms | previous_terms))
        if current_terms and previous_terms else 0.0
    )
    action_claim = any(term in response for term in (
        "bhej diya", "send kar diya", "forward kar diya", "deliver kar diya",
        "call kar diya", "message kar diya",
    ))
    action_request = any(term in user for term in (
        "bata", "message", "bhej", "send", "forward", "call",
    ))
    return {
        "repeated_greeting": turn_index > 1 and greeting,
        "caller_correction": correction,
        "repeated_clarification": previous_question and current_question and overlap >= 0.55,
        "privacy_boundary_applicable": privacy_request,
        "privacy_boundary_compliant": (not privacy_request) or privacy_refusal,
        # This is intentionally named a proxy: tool execution evidence is not
        # available in this pure text evaluator.
        "fabricated_action_proxy": action_claim and not action_request,
    }


def summarize_turns(records: Iterable[dict[str, Any]], silent_after_ms: int = 2500) -> dict[str, Any]:
    """Summarize metadata-only live or replayed voice-turn records.

    ``first_answer_audio_ms`` is a server-to-telephony hand-off proxy. A
    telephone provider does not report actual handset playback, so this module
    never labels it as caller-heard audio.
    """
    all_rows = [record.get("metrics", record) for record in records]
    # Version 1 mixed queueing with model TTFT and could anchor a response to
    # a stale VAD boundary.  Do not combine it with corrected measurements.
    rows = [row for row in all_rows if row.get("evaluation_schema_version") == CURRENT_SCHEMA_VERSION]
    completed = [row for row in rows if row.get("evaluation_finalized")]
    def values(name: str) -> list[float]:
        return [float(row[name]) for row in completed if isinstance(row.get(name), (int, float))]

    def distribution(name: str) -> dict[str, float | int | None]:
        observed = values(name)
        return {"count": len(observed), "p50_ms": percentile(observed, 50), "p95_ms": percentile(observed, 95), "p99_ms": percentile(observed, 99)}

    def value_distribution(name: str) -> dict[str, float | int | None]:
        observed = values(name)
        return {"count": len(observed), "p50": percentile(observed, 50), "p95": percentile(observed, 95), "p99": percentile(observed, 99)}

    def rate(name: str, *, applicable: str | None = None) -> dict[str, float | int | None]:
        population = [row for row in completed if applicable is None or row.get(applicable) is True]
        labelled = [row for row in population if isinstance(row.get(name), bool)]
        positives = sum(row[name] is True for row in labelled)
        return {"count": positives, "labelled": len(labelled), "rate": round(positives / len(labelled), 4) if labelled else None}
    silent = [
        row for row in completed
        if row.get("silent_turn") is True
        or not isinstance(row.get("first_answer_audio_ms"), (int, float))
        or row["first_answer_audio_ms"] > silent_after_ms
    ]
    annotated = [row for row in completed if isinstance(row.get("reference_turn_end_ms"), (int, float))]
    premature = [
        row for row in annotated
        if isinstance(row.get("first_answer_audio_at_ms"), (int, float))
        and row["first_answer_audio_at_ms"] < row["reference_turn_end_ms"]
    ]
    barge_attempts = [row for row in completed if isinstance(row.get("barge_in_detected_at_ms"), (int, float))]
    barge_stop_latencies = [
        row["barge_in_stop_latency_ms"]
        for row in barge_attempts
        if isinstance(row.get("barge_in_stop_latency_ms"), (int, float))
    ]

    return {
        "definition": {
            "measurement_schema_version": CURRENT_SCHEMA_VERSION,
            "first_answer_audio": "first answer media frame handed to VoiceLink after commit; excludes latency cues and cannot prove handset playback",
            "first_cue_audio": "first latency-cue frame handed to VoiceLink; never counted as an answer",
            "provider_ttft": "provider-reported request-to-first-token latency",
            "answer_decode_ttft": "request-to-first decoded answer character, including any structured-output prefix",
            "turn_queue_delay": "time from committed transcript to its turn beginning processing",
            "silent_turn": f"no first answer frame within {silent_after_ms} ms after a committed user turn",
            "premature_turn": "requires an independently labelled reference_turn_end_ms; not inferred from VAD alone",
            "barge_in_stop": "time from detected caller speech while agent output is active to the pipeline's clear/stop hand-off",
            "estimated_tokens": "character-count estimate; provider token counts are exposed separately when available",
        },
        "turns": {
            "observed": len(rows), "completed": len(completed),
            "excluded_legacy_measurements": len(all_rows) - len(rows),
        },
        "endpointing": {name: distribution(name) for name in (
            "speech_end_to_commit_ms", "provider_final_wait_ms", "vad_semantic_wait_ms", "turn_queue_delay_ms",
        )},
        "prompt_formation": {name: distribution(name) for name in (
            "account_context_ms", "router_ms", "context_assembly_ms", "retrieval_ms", "prompt_build_ms",
        )} | {
            "prompt_characters": value_distribution("prompt_characters"),
            "prompt_token_estimate": value_distribution("prompt_token_estimate"),
            "prompt_versions": dict(Counter(str(row["prompt_version"]) for row in completed if row.get("prompt_version"))),
        },
        "llm": {name: distribution(name) for name in (
            "llm_request_setup_ms", "provider_ttft_ms", "answer_decode_ttft_ms", "llm_total_stream_ms", "e2e_first_answer_text_ms",
        )} | {
            "input_tokens": value_distribution("input_tokens"),
            "output_tokens": value_distribution("output_tokens"),
            "output_token_estimate": value_distribution("output_token_estimate"),
        },
        "tts": {name: distribution(name) for name in (
            "text_to_first_fragment_ms", "tts_connect_ms", "tts_ttfa_ms",
        )},
        "telephony": {name: distribution(name) for name in (
            "first_cue_audio_ms", "first_answer_audio_ms", "first_frame_send_ms",
            "outbound_frame_gap_p50_ms", "outbound_frame_gap_p95_ms", "outbound_frame_gap_max_ms",
            "outbound_audio_duration_ms",
        )},
        "first_answer_audio_proxy": distribution("first_answer_audio_ms"),
        "provider_ttft": distribution("provider_ttft_ms"),
        "turn_queue_delay": distribution("turn_queue_delay_ms"),
        "silent_turn_rate": round(len(silent) / len(completed), 4) if completed else None,
        "silent_turns": len(silent),
        "premature_turn_rate": round(len(premature) / len(annotated), 4) if annotated else None,
        "premature_turns": len(premature),
        "premature_turns_annotated": len(annotated),
        "barge_in_attempts": len(barge_attempts),
        "barge_in_stop_latency": {"count": len(barge_stop_latencies), "p50_ms": percentile(barge_stop_latencies, 50), "p95_ms": percentile(barge_stop_latencies, 95), "p99_ms": percentile(barge_stop_latencies, 99)},
        "reliability": {name: rate(name) for name in (
            "json_parse_failure", "llm_failure", "tts_failure", "interrupted_turn", "partial_delivery",
        )} | {
            "tts_zero_audio_retry_count": sum(int(row.get("tts_zero_audio_retry_count", 0)) for row in completed),
            "tts_reconnect_count": sum(int(row.get("tts_reconnect_count", 0)) for row in completed),
        },
        "conversation_quality": {
            "repeated_greeting": rate("repeated_greeting"),
            "fabricated_action_proxy": rate("fabricated_action_proxy"),
            "repeated_clarification": rate("repeated_clarification"),
            "privacy_boundary_non_compliance": rate("privacy_boundary_non_compliance", applicable="privacy_boundary_applicable"),
            "caller_correction": rate("caller_correction"),
        },
        "audio_stt": {name: rate(name) for name in (
            "false_endpoint", "premature_turn", "missed_turn", "noise_trigger",
        )},
    }


def summarize_wer(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate labelled STT samples without depending on a provider SDK."""
    samples = [row for row in rows if isinstance(row.get("reference_text"), str) and isinstance(row.get("hypothesis_text"), str)]
    scores = [word_error_rate(row["reference_text"], row["hypothesis_text"]) for row in samples]
    words = sum(int(score["reference_words"]) for score in scores)
    errors = sum(int(score["errors"]) for score in scores)
    return {"samples": len(scores), "reference_words": words, "errors": errors, "wer": round(errors / words, 4) if words else None}
