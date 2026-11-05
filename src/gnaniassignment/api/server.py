"""Provider-neutral HTTP entrypoint for the personal phone assistant.

Telephone vendors should send lifecycle/transcript webhooks here. Their live
audio WebSocket protocol remains an adapter concern; this module deliberately
does not pretend a plain HTTP webhook can carry a bidirectional phone call.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gnaniassignment.app.config import load_environment
from gnaniassignment.app.container import create_session
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.prompts.tree import PostgresPromptTreeEditor, PostgresPromptTreeStore, PromptPlan


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    customer_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)


class CallWebhook(BaseModel):
    """Normalized event emitted by a provider-specific telephony adapter."""

    event: str = Field(pattern=r"^(call\.started|transcript\.final|call\.ended)$")
    call_id: str = Field(min_length=1, max_length=256)
    caller_id: str = Field(min_length=1, max_length=256)
    transcript: str | None = Field(default=None, max_length=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptNodeCreate(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)
    priority: int = Field(default=100, ge=-10_000, le=10_000)
    condition: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RuntimeEvents:
    """In-process fan-out for the operations dashboard.

    PostgreSQL remains the durable source for calls and classifier feedback;
    this only delivers the immediate, live view for this API process.
    """

    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    loop: asyncio.AbstractEventLoop | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def publish(self, kind: str, **data: Any) -> None:
        self.loop = asyncio.get_running_loop()
        event = {"kind": kind, "at": datetime.now(UTC).isoformat(), **data}
        async with self.lock:
            for queue in tuple(self.subscribers):
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(event)

    def publish_from_thread(self, kind: str, **data: Any) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(asyncio.create_task, self.publish(kind, **data))

    async def stream(self) -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        async with self.lock:
            self.subscribers.add(queue)
        try:
            yield b"retry: 3000\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"event: runtime\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                except TimeoutError:
                    yield b": keepalive\n\n"
        finally:
            async with self.lock:
                self.subscribers.discard(queue)


@dataclass
class SessionRegistry:
    """One-process session registry; replace with Redis ownership for replicas."""

    sessions: dict[str, TurnSession] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    events: RuntimeEvents = field(default_factory=RuntimeEvents)

    async def get_or_create(self, session_id: str, customer_id: str) -> TurnSession:
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = create_session(
                    customer_id=customer_id,
                    shadow_observer=lambda prediction: self._shadow_complete(session_id, prediction),
                    shadow_error_observer=lambda error: self._shadow_failed(session_id, error),
                )
                self.sessions[session_id] = session
            return session

    def _shadow_complete(self, session_id: str, prediction: dict[str, Any]) -> None:
        decision = prediction.get("answers", {}).get("intent", {})
        self.events.publish_from_thread(
            "classifier.completed",
            session_id=session_id,
            label=decision.get("choice", "unknown"),
            confidence=decision.get("confidence"),
            pipeline="minilm-shadow",
        )

    def _shadow_failed(self, session_id: str, error: Exception) -> None:
        # Keep raw stack traces on the server; the dashboard only needs a safe
        # actionable summary to explain why no candidate was produced.
        self.events.publish_from_thread(
            "classifier.failed",
            session_id=session_id,
            pipeline="minilm-shadow",
            message=str(error)[:300],
        )

    async def end(self, session_id: str) -> dict[str, object] | None:
        async with self.lock:
            session = self.sessions.pop(session_id, None)
        return await asyncio.to_thread(session.finalize_call) if session is not None else None


def _require_api_key(value: str | None) -> None:
    configured = os.getenv("AGENT_API_KEY")
    if not configured:
        if os.getenv("AGENT_ENVIRONMENT", "development").lower() == "production":
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AGENT_API_KEY is required in production")
        return
    if value is None or not hmac.compare_digest(value, configured):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _verify_webhook(raw_body: bytes, signature: str | None) -> None:
    secret = os.getenv("WEBHOOK_SIGNING_SECRET")
    if not secret:
        if os.getenv("AGENT_ENVIRONMENT", "development").lower() == "production":
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="WEBHOOK_SIGNING_SECRET is required in production")
        return
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    if signature is None or not hmac.compare_digest(signature.removeprefix("sha256="), expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature")


def _prompt_plan_payload(plan: PromptPlan | None) -> dict[str, Any]:
    runtime_enabled = os.getenv("PROMPT_TREE_ENABLED", "false").lower() == "true"
    if plan is None:
        return {"configured": False, "runtime_enabled": runtime_enabled, "version": None, "nodes": []}
    return {
        "configured": True,
        "runtime_enabled": runtime_enabled,
        "tenant_id": plan.tenant_id,
        "agent_id": plan.agent_id,
        "version": plan.version,
        "nodes": [
            {
                "id": node.node_id,
                "instruction": node.instruction,
                "priority": node.priority,
                "condition": node.condition,
                "parent_id": node.parent_id,
            }
            for node in plan.nodes
        ],
    }


def create_app(registry: SessionRegistry | None = None) -> FastAPI:
    load_environment()
    sessions = registry or SessionRegistry()
    app = FastAPI(title="Personal Phone Assistant API", version="0.1.0")
    origins = [origin.strip() for origin in os.getenv("DASHBOARD_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "time": datetime.now(UTC).isoformat()}

    @app.get("/v1/dashboard/events")
    async def dashboard_events(x_api_key: str | None = Header(default=None)) -> StreamingResponse:
        _require_api_key(x_api_key)
        return StreamingResponse(
            sessions.events.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/dashboard/prompt-plan")
    async def dashboard_prompt_plan(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        try:
            plan = PostgresPromptTreeStore().load_active(os.getenv("TENANT_ID", "default"), "generic")
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return _prompt_plan_payload(plan)

    @app.post("/v1/dashboard/prompt-plan/nodes", status_code=status.HTTP_201_CREATED)
    async def dashboard_add_prompt_node(payload: PromptNodeCreate, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        try:
            plan = PostgresPromptTreeEditor().append_node(
                os.getenv("TENANT_ID", "default"),
                "generic",
                payload.instruction,
                payload.priority,
                payload.condition,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        # Evict process and optional Redis cache immediately, so new calls use
        # the published version instead of waiting for the plan TTL.
        from gnaniassignment.app.container import create_prompt_resolver
        resolver = create_prompt_resolver()
        if resolver is not None:
            resolver.invalidate(plan.tenant_id, plan.agent_id)
        await sessions.events.publish("prompt.published", version=plan.version, nodes=len(plan.nodes))
        return _prompt_plan_payload(plan)

    @app.post("/v1/turn")
    async def turn(payload: TurnRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
        _require_api_key(x_api_key)
        session_id = payload.session_id or f"api:{payload.customer_id}"
        session = await sessions.get_or_create(session_id, payload.customer_id)
        await sessions.events.publish("transcript.received", session_id=session_id, customer_id=payload.customer_id, text=payload.text)
        if os.getenv("SHADOW_INTENT_ENABLED", "false").lower() == "true":
            await sessions.events.publish("classifier.queued", session_id=session_id, pipeline="minilm-shadow")
        result = await asyncio.to_thread(session.ask, payload.text)
        await sessions.events.publish("agent.responded", session_id=session_id, intent=result["intent"], text=result["spoken"])
        return {"session_id": session_id, "answer": result["spoken"], "intent": result["intent"]}

    @app.post("/v1/webhooks/call")
    async def call_webhook(request: Request, x_webhook_signature: str | None = Header(default=None)) -> dict[str, object]:
        raw_body = await request.body()
        _verify_webhook(raw_body, x_webhook_signature)
        try:
            payload = CallWebhook.model_validate_json(raw_body)
        except Exception as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid call webhook payload") from error

        if payload.event == "call.started":
            await sessions.get_or_create(payload.call_id, payload.caller_id)
            await sessions.events.publish("call.started", session_id=payload.call_id, customer_id=payload.caller_id)
            return {"accepted": True, "call_id": payload.call_id}
        if payload.event == "call.ended":
            judgment = await sessions.end(payload.call_id)
            await sessions.events.publish("call.ended", session_id=payload.call_id)
            return {"accepted": True, "call_id": payload.call_id, "judgment": judgment}
        if not payload.transcript or not payload.transcript.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="transcript.final requires transcript")
        session = await sessions.get_or_create(payload.call_id, payload.caller_id)
        await sessions.events.publish("transcript.received", session_id=payload.call_id, customer_id=payload.caller_id, text=payload.transcript)
        if os.getenv("SHADOW_INTENT_ENABLED", "false").lower() == "true":
            await sessions.events.publish("classifier.queued", session_id=payload.call_id, pipeline="minilm-shadow")
        result = await asyncio.to_thread(session.ask, payload.transcript)
        await sessions.events.publish("agent.responded", session_id=payload.call_id, intent=result["intent"], text=result["spoken"])
        return {"accepted": True, "call_id": payload.call_id, "answer": result["spoken"], "intent": result["intent"]}

    return app


app = create_app()
