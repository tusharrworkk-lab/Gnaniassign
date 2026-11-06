"""Measure local Laya inference and prove the device executing its parameters."""

from __future__ import annotations

import argparse
import statistics
import time

from gnaniassignment.app.config import load_environment
from gnaniassignment.llm.local_laya import LocalLayaRouter
from gnaniassignment.llm.prompts.intents import QUESTIONS


def _synchronize_if_cuda(router: LocalLayaRouter) -> None:
    if str(router.device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(router.device)


def _same_execution_device(router_device: object, parameter_device: object) -> bool:
    """Treat CUDA's implicit default device and cuda:0 as equivalent."""
    def parse(device: object) -> tuple[str, int | None]:
        raw = str(device).strip().lower()
        device_type, separator, ordinal = raw.partition(":")
        if not separator or not ordinal:
            return device_type, None
        try:
            return device_type, int(ordinal)
        except ValueError:
            return device_type, None

    router_type, router_index = parse(router_device)
    parameter_type, parameter_index = parse(parameter_device)
    if router_type != parameter_type:
        return False
    if router_type != "cuda":
        return str(router_device).strip().lower() == str(parameter_device).strip().lower()
    # This app selects the default CUDA device (cuda) or an explicit ordinal.
    # `cuda` resolves to cuda:0 unless the caller deliberately set an ordinal.
    return (router_index if router_index is not None else 0) == (parameter_index if parameter_index is not None else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one warmed local Laya router")
    parser.add_argument("--runs", type=int, default=10, help="Measured warm predictions (default: 10)")
    parser.add_argument("--backend", choices=("eager", "compile", "auto", "tilelang"), help="Override LOCAL_LAYA_BACKEND for this benchmark")
    parser.add_argument("--profile", action="store_true", help="Print CPU and CUDA operation time for one warmed prediction")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    load_environment()
    router = LocalLayaRouter(backend=args.backend)
    model_device = next(router.agent.model.parameters()).device
    print(f"laya_device: {router.device}")
    print(f"model_parameter_device: {model_device}")
    print(f"laya_backend: {router.backend}")
    print(f"laya_dtype: {getattr(router.agent, 'dtype', 'unknown')}")
    print(f"load_and_warmup_ms: {router.load_ms:.0f}")
    if not _same_execution_device(router.device, model_device):
        raise RuntimeError("Laya router device and model parameter device differ")

    state = {"current_query": "Where do you operate?", "recent_conversation": []}
    values: list[float] = []
    last_result = None
    for _ in range(args.runs):
        _synchronize_if_cuda(router)
        started = time.perf_counter()
        last_result = router.predict(state, QUESTIONS)
        _synchronize_if_cuda(router)
        values.append((time.perf_counter() - started) * 1000)
    print("warm_predict_ms: " + ", ".join(f"{value:.0f}" for value in values))
    print(f"median_warm_predict_ms: {statistics.median(values):.0f}")
    if isinstance(last_result, dict):
        usage = last_result.get("usage") or {}
        if "input_tokens" in usage:
            print(f"laya_input_tokens: {usage['input_tokens']}")
    if str(router.device).startswith("cuda"):
        import torch

        print(f"cuda_memory_allocated_mb: {torch.cuda.memory_allocated(router.device) / 1024**2:.0f}")
        print(f"cuda_memory_reserved_mb: {torch.cuda.memory_reserved(router.device) / 1024**2:.0f}")
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                router.predict(state, QUESTIONS)
                torch.cuda.synchronize(router.device)
            print("cuda_profile_top_operations:")
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
