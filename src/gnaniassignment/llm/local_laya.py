"""In-process Laya intent decisions with explicit device placement."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any


def cuda_runtime_report() -> dict[str, object]:
    """Return CUDA facts without loading the Laya checkpoint."""
    import torch

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        report["device_name"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        report["compiled_architectures"] = torch.cuda.get_arch_list()
    return report


def require_requested_cuda(device: str) -> None:
    """Reject a requested CUDA device when PyTorch cannot use it.

    Laya itself intentionally falls back to CPU. That is useful for notebooks, but
    incorrect for this latency-sensitive production path: a GPU configuration must
    either use CUDA or fail with an actionable error.
    """
    if not device.strip().lower().startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "LOCAL_LAYA_DEVICE requests CUDA, but PyTorch cannot access an NVIDIA GPU. "
            f"Installed PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}. "
            "Fix the NVIDIA driver/runtime first (nvidia-smi must succeed); in Docker pass --gpus all, "
            "and in WSL install the NVIDIA Windows driver. Set LOCAL_LAYA_DEVICE=cpu only when CPU "
            "inference is intentional."
        )
    if ":" in device:
        try:
            index = int(device.rsplit(":", 1)[1])
        except ValueError as exc:
            raise ValueError("LOCAL_LAYA_DEVICE must be cuda or cuda:<non-negative index>") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_LAYA_DEVICE={device} is unavailable; {torch.cuda.device_count()} CUDA device(s) visible")


@lru_cache(maxsize=2)
def _load_agent(model_id: str, device: str, backend: str) -> Any:
    # Keep the checkpoint resident across conversation sessions in one process.
    import laya

    from gnaniassignment.llm.prompts.intents import QUESTIONS

    # Load stock eager first. Laya's backend selection normally degrades to eager
    # with a warning when an accelerated backend is unavailable; that is safe for
    # production but would make an acceleration benchmark misleading.
    agent = laya.load(model_id, device=device, backend="eager")
    if device.lower().startswith("cuda") and getattr(agent.device, "type", None) != "cuda":
        raise RuntimeError("Laya did not remain on CUDA; refusing its silent CPU fallback")
    if backend != "eager":
        try:
            agent.set_backend(backend, strict=True)
        except Exception as exc:
            target_hint = ""
            if backend == "tilelang" and "No registered target detector" in str(exc):
                target_hint = (
                    " For a Turing GTX 1650 Ti, retry with "
                    "TILELANG_DEFAULT_TARGET='{" + '\"kind\":\"cuda\",\"arch\":\"sm_75\"' + "}'."
                )
            raise RuntimeError(
                f"Requested LOCAL_LAYA_BACKEND={backend!r}, but Laya could not activate it. "
                "The benchmark must not continue on eager." + target_hint
            ) from exc
    # Pay CUDA initialization and the first forward pass at startup, not on a user's turn.
    agent.predict({"current_query": "What does the product do?", "recent_conversation": []}, QUESTIONS)
    return agent


class LocalLayaRouter:
    def __init__(self, model_id: str | None = None, device: str | None = None, backend: str | None = None, agent: Any = None) -> None:
        self.model_id = model_id or os.getenv("LOCAL_LAYA_MODEL") or "convaiinnovations/laya"
        self.device = device or os.getenv("LOCAL_LAYA_DEVICE") or "cuda"
        # Eager CUDA is the stable low-batch path. Laya's auto backend enables
        # torch.compile, whose compile cost usually loses for short intent prompts.
        self.backend = backend or os.getenv("LOCAL_LAYA_BACKEND") or "eager"
        started = time.perf_counter()
        if agent is None:
            require_requested_cuda(self.device)
            self.agent = _load_agent(self.model_id, self.device, self.backend)
            self.device = str(self.agent.device)
            self.backend = getattr(self.agent, "backend", self.backend)
        else:
            self.agent = agent
        self.load_ms = (time.perf_counter() - started) * 1000
        self.last_ms = 0.0

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        if model is not None and model != self.model_id:
            raise ValueError("A local Laya router cannot switch checkpoints after loading")
        started = time.perf_counter()
        try:
            result = self.agent.predict(state, questions)
            if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
                raise RuntimeError("Local Laya response did not contain typed answers")
            return result
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
