"""Verify local CUDA availability before starting the Laya intent router."""

from __future__ import annotations

import argparse

from gnaniassignment.app.config import load_environment
from gnaniassignment.llm.local_laya import LocalLayaRouter, cuda_runtime_report, require_requested_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description="Show CUDA/Laya device status")
    parser.add_argument("--load", action="store_true", help="Also load and warm the Laya checkpoint")
    args = parser.parse_args()
    load_environment()
    report = cuda_runtime_report()
    for key, value in report.items():
        print(f"{key}: {value}")
    router = LocalLayaRouter if args.load else None
    if args.load:
        instance = router()
        print(f"laya_device: {instance.device}")
        print(f"laya_backend: {instance.backend}")
        print(f"laya_load_ms: {instance.load_ms:.0f}")
    else:
        import os

        try:
            require_requested_cuda(os.getenv("LOCAL_LAYA_DEVICE", "cuda"))
        except RuntimeError as exc:
            parser.exit(2, f"CUDA validation: FAILED\n{exc}\n")
        print("CUDA validation: passed")


if __name__ == "__main__":
    main()
