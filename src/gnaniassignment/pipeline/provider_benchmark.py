"""Compare Vercel AI Gateway routing and streaming latency without changing app routing."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import requests

from gnaniassignment.app.config import load_environment
from gnaniassignment.llm.chat import DEFAULT_GATEWAY_URL, DEFAULT_SUBAGENT_MODEL
from gnaniassignment.llm.prompts.enquiry_agents import PROMPTS

DEFAULT_PROVIDERS = ("auto", "bedrock", "novita")


def routing_from_event(event: dict) -> dict:
    """Read Vercel's routing metadata when present on a stream event."""
    choices = event.get("choices") or []
    sources = [event]
    if choices:
        sources.extend((choices[0], choices[0].get("delta") or {}, choices[0].get("message") or {}))
    for source in sources:
        metadata = source.get("provider_metadata") or source.get("providerMetadata") or {}
        gateway = metadata.get("gateway") or {}
        routing = gateway.get("routing") or {}
        if routing:
            return routing
    return {}


def provider_from_event(event: dict) -> str | None:
    return routing_from_event(event).get("finalProvider")


def measure(session: requests.Session, url: str, key: str, model: str, query: str, provider: str) -> dict:
    user_prompt = (
        "Approved knowledge:\nNo approved knowledge has been loaded yet.\n\n"
        f"Relevant conversation:\n\nCurrent user query:\n{query}"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPTS["product_what"]},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
        "stream": True,
    }
    if provider != "auto":
        body["providerOptions"] = {"gateway": {"only": [provider]}}

    started = time.perf_counter()
    with session.post(
        f"{url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
        stream=True,
        timeout=20,
    ) as response:
        headers_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        first_event_ms = None
        first_text_ms = None
        actual_provider = None
        provider_attempt_ms = None
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if first_event_ms is None:
                first_event_ms = (time.perf_counter() - started) * 1000
            routing = routing_from_event(event)
            actual_provider = routing.get("finalProvider") or actual_provider
            attempts = routing.get("modelAttempts") or []
            for attempt in attempts:
                for provider_attempt in attempt.get("providerAttempts") or []:
                    if provider_attempt.get("success"):
                        begin = provider_attempt.get("startTime")
                        end = provider_attempt.get("endTime")
                        if isinstance(begin, (int, float)) and isinstance(end, (int, float)):
                            provider_attempt_ms = end - begin
            choices = event.get("choices") or []
            if first_text_ms is None and choices and (choices[0].get("delta") or {}).get("content"):
                first_text_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
    return {
        "requested": provider,
        "actual": actual_provider or "not exposed",
        "headers_ms": headers_ms,
        "first_event_ms": first_event_ms,
        "first_text_ms": first_text_ms,
        "provider_attempt_ms": provider_attempt_ms,
        "total_ms": total_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare auto and pinned Vercel providers using one prompt")
    parser.add_argument("--runs", type=int, default=2, help="Runs per provider (default: 2)")
    parser.add_argument("--providers", nargs="+", default=list(DEFAULT_PROVIDERS), help="Use auto or provider slugs")
    parser.add_argument("--query", default="What does the product do?")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    load_environment()
    key = os.getenv("AI_GATEWAY_API_KEY")
    if not key:
        parser.error("Set AI_GATEWAY_API_KEY in .env or your environment")
    url = os.getenv("SUBAGENT_GATEWAY_URL") or DEFAULT_GATEWAY_URL
    model = os.getenv("SUBAGENT_MODEL") or DEFAULT_SUBAGENT_MODEL
    measurements = []
    with requests.Session() as session:
        for run in range(args.runs):
            for provider in args.providers:
                try:
                    result = measure(session, url, key, model, args.query, provider)
                except (requests.RequestException, ValueError) as exc:
                    print(f"run {run + 1} {provider}: {exc}")
                    continue
                measurements.append(result)
                event_ms = f"{result['first_event_ms']:.0f}" if result["first_event_ms"] is not None else "n/a"
                text_ms = f"{result['first_text_ms']:.0f}" if result["first_text_ms"] is not None else "n/a"
                attempt_ms = f"{result['provider_attempt_ms']:.0f}" if result["provider_attempt_ms"] is not None else "n/a"
                print(
                    f"run {run + 1} {provider:8} actual={result['actual']:12} "
                    f"headers={result['headers_ms']:.0f} ms "
                    f"event={event_ms} ms "
                    f"text={text_ms} ms "
                    f"provider_attempt={attempt_ms} ms "
                    f"total={result['total_ms']:.0f} ms"
                )
    for provider in args.providers:
        values = [r["first_text_ms"] for r in measurements if r["requested"] == provider and r["first_text_ms"] is not None]
        if values:
            print(f"{provider} median first text: {statistics.median(values):.0f} ms ({len(values)} requests)")


if __name__ == "__main__":
    main()
