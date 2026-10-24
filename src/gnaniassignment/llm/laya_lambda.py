"""Private AWS Lambda adapter for a laya.cpp System One function."""

from __future__ import annotations

import json
import os
import time
from typing import Any


DEFAULT_FUNCTION_NAME = "gnani-laya-multilingual"


class LayaLambdaRouter:
    """Invoke Laya privately with IAM, avoiding a public endpoint or API Gateway."""

    def __init__(self, function_name: str | None = None, client: Any | None = None) -> None:
        self.function_name = function_name or os.getenv("LAYA_LAMBDA_FUNCTION") or DEFAULT_FUNCTION_NAME
        self.region = os.getenv("LAYA_LAMBDA_REGION", os.getenv("AWS_REGION", "ap-south-1"))
        self.timeout_seconds = float(os.getenv("LAYA_LAMBDA_TIMEOUT_SECONDS", "15"))
        self.last_ms = 0.0
        if client is not None:
            self.client = client
            return
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("Install Lambda router support with: uv sync --extra bedrock") from error
        self.client = boto3.client(
            "lambda", region_name=self.region,
            config=Config(read_timeout=self.timeout_seconds, connect_timeout=2, retries={"max_attempts": 0}),
        )

    def predict(self, state: str | dict[str, Any], questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = {"model": model or "laya", "state": state, "questions": questions}
            response = self.client.invoke(
                FunctionName=self.function_name, InvocationType="RequestResponse", Payload=json.dumps(payload).encode("utf-8")
            )
            envelope = json.loads(response["Payload"].read())
            status = int(envelope.get("statusCode", 502))
            body = json.loads(envelope.get("body", "{}"))
            if status != 200:
                raise RuntimeError(body.get("error", f"Laya Lambda returned status {status}"))
            if not isinstance(body.get("answers"), dict):
                raise RuntimeError("Laya Lambda response did not contain typed answers")
            return body
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000
