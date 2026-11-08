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
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from gnaniassignment.app.config import load_environment
from gnaniassignment.app.container import create_session
from gnaniassignment.orchestration.session import TurnSession
from gnaniassignment.prompts.tree import PostgresPromptTreeEditor, PostgresPromptTreeStore, PromptPlan


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    customer_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)


class HarnessConversationCreate(BaseModel):
    """Start an isolated, text-only agent conversation for evaluation."""

    customer_id: str = Field(default="harness", min_length=1, max_length=256)


class HarnessConversationTurn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


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


class PlaneCommentDraftCreate(BaseModel):
    """A caller-proposed comment. This endpoint never writes to Plane."""

    requested_by: str = Field(min_length=1, max_length=256)
    project_id: str = Field(min_length=1, max_length=256)
    work_item_id: str = Field(min_length=1, max_length=256)
    comment: str = Field(min_length=1, max_length=10_000)


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
    turn_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
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
                self.turn_locks[session_id] = asyncio.Lock()
            return session

    async def get(self, session_id: str) -> TurnSession | None:
        """Return an existing session without accidentally creating one."""
        async with self.lock:
            return self.sessions.get(session_id)

    async def run_turn(self, session_id: str, text: str) -> dict[str, Any] | None:
        """Serialize turns in a session so concurrent harness requests cannot
        reorder the agent's history or working memory.
        """
        async with self.lock:
            session = self.sessions.get(session_id)
            turn_lock = self.turn_locks.get(session_id)
        if session is None or turn_lock is None:
            return None
        async with turn_lock:
            return await asyncio.to_thread(session.ask, text)

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
            session = self.sessions.get(session_id)
            turn_lock = self.turn_locks.get(session_id)
        if session is None or turn_lock is None:
            return None
        async with turn_lock:
            async with self.lock:
                # A completed session cannot be resurrected by a late request.
                if self.sessions.pop(session_id, None) is None:
                    return None
                self.turn_locks.pop(session_id, None)
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


def _authorize_voicelink_websocket(token: str | None) -> bool:
    """Authenticate the telephony media stream before accepting the upgrade.

    VoiceLink documents the WebSocket URL but no request-header authentication
    field. A secret query parameter works with a standard WSS URL and keeps
    this provider adapter fail-closed in production.
    """
    configured = os.getenv("VOICELINK_WS_TOKEN")
    if not configured:
        return os.getenv("AGENT_ENVIRONMENT", "development").lower() != "production"
    return token is not None and hmac.compare_digest(token, configured)


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


def _harness_conversation_payload(conversation_id: str, session: TurnSession) -> dict[str, Any]:
    """Expose only the evaluation state needed to inspect dialogue behaviour."""
    return {
        "conversation_id": conversation_id,
        "customer_id": session.customer_id,
        "working_memory": dict(session.working_memory),
        "turns": [
            {
                "turn_number": index,
                "user": turn.query,
                "assistant": turn.answer,
                "intent": turn.intent,
                "topic": turn.topic,
            }
            for index, turn in enumerate(session.call_turns, start=1)
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

    def calendar_enabled() -> bool:
        return os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() in {"1", "true", "yes", "on"}

    def calendar_services():
        from gnaniassignment.google_calendar import GoogleCalendarClient, GoogleCalendarStore
        return GoogleCalendarStore(), GoogleCalendarClient()

    async def refresh_calendar_once() -> None:
        if not calendar_enabled():
            return
        from gnaniassignment.google_calendar import CalendarIntegrationError, GoogleCalendarSettings, refresh_current_event
        try:
            store, client = calendar_services()
            settings = GoogleCalendarSettings.from_environment()
            snapshot = await asyncio.to_thread(refresh_current_event, store, client, settings.owner_id)
            await sessions.events.publish("calendar.presence_refreshed", status=snapshot.get("status"))
        except (CalendarIntegrationError, RuntimeError, ValueError) as error:
            # A disconnected calendar must not affect inbound phone calls.
            await sessions.events.publish("calendar.refresh_failed", message=str(error)[:240])

    async def calendar_refresh_loop() -> None:
        while True:
            await refresh_calendar_once()
            await asyncio.sleep(max(30, int(os.getenv("GOOGLE_CALENDAR_REFRESH_SECONDS", "60"))))

    @app.on_event("startup")
    async def begin_calendar_refresh() -> None:
        if calendar_enabled():
            app.state.calendar_refresh_task = asyncio.create_task(calendar_refresh_loop())

    @app.on_event("shutdown")
    async def stop_calendar_refresh() -> None:
        task = getattr(app.state, "calendar_refresh_task", None)
        if task is not None:
            task.cancel()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "time": datetime.now(UTC).isoformat()}

    @app.websocket("/v1/media/voicelink")
    async def voicelink_media(websocket: WebSocket) -> None:
        """VoiceLink's documented bidirectional A-law media stream."""
        if not _authorize_voicelink_websocket(websocket.query_params.get("token")):
            # Do not accept an unauthenticated upgrade; 1008 is the standard
            # WebSocket policy-violation close code.
            await websocket.close(code=1008)
            return
        from gnaniassignment.telephony.voicelink import handle_voicelink_websocket
        try:
            await handle_voicelink_websocket(websocket, sessions.events)
        except WebSocketDisconnect:
            return

    @app.get("/v1/dashboard/events")
    async def dashboard_events(x_api_key: str | None = Header(default=None)) -> StreamingResponse:
        _require_api_key(x_api_key)
        return StreamingResponse(
            sessions.events.stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/dashboard/latency")
    async def dashboard_latency(limit: int = 100, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        from gnaniassignment.telephony.latency import CallLatencyStore
        return {"calls": await asyncio.to_thread(CallLatencyStore().recent, max(1, min(limit, 500)))}

    @app.get("/v1/integrations/google-calendar/connect")
    async def google_calendar_connect(x_api_key: str | None = Header(default=None)) -> dict[str, str]:
        """Return an OAuth URL after dashboard API-key authentication."""
        _require_api_key(x_api_key)
        if not calendar_enabled():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Google Calendar integration is disabled")
        from gnaniassignment.google_calendar import GoogleCalendarClient, GoogleCalendarSettings
        try:
            settings = GoogleCalendarSettings.from_environment()
            return {"authorization_url": GoogleCalendarClient(settings).authorization_url(settings.owner_id)}
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    @app.get("/v1/integrations/google-calendar/callback", response_class=HTMLResponse)
    async def google_calendar_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> HTMLResponse:
        """Google's browser redirect endpoint; it never exposes a token."""
        if error:
            return HTMLResponse("<h2>Google Calendar was not connected.</h2><p>You can close this tab and try again from the dashboard.</p>", status_code=status.HTTP_400_BAD_REQUEST)
        if not code or not state:
            return HTMLResponse("<h2>Calendar connection failed.</h2><p>Missing authorization response.</p>", status_code=status.HTTP_400_BAD_REQUEST)
        from gnaniassignment.google_calendar import CalendarIntegrationError, GoogleCalendarClient, GoogleCalendarSettings, GoogleCalendarStore, refresh_current_event
        try:
            settings = GoogleCalendarSettings.from_environment()
            client = GoogleCalendarClient(settings)
            owner_id = client.verify_state(state)
            result = await asyncio.to_thread(client.exchange_code, code)
            store = GoogleCalendarStore(settings)
            await asyncio.to_thread(store.save_refresh_token, owner_id, result["refresh_token"], result.get("scope"))
            await asyncio.to_thread(refresh_current_event, store, client, owner_id)
            await sessions.events.publish("calendar.connected", owner_id=owner_id)
        except (CalendarIntegrationError, RuntimeError, ValueError, KeyError) as exc:
            return HTMLResponse("<h2>Calendar connection failed.</h2><p>Please return to the dashboard and start again.</p>", status_code=status.HTTP_400_BAD_REQUEST)
        return HTMLResponse("<h2>Google Calendar connected.</h2><p>Your current availability is now ready for future calls. You can close this tab.</p>")

    @app.get("/v1/integrations/google-calendar/status")
    async def google_calendar_status(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        if not calendar_enabled():
            return {"enabled": False, "connected": False}
        from gnaniassignment.google_calendar import GoogleCalendarSettings
        try:
            settings = GoogleCalendarSettings.from_environment()
            store, _ = calendar_services()
            snapshot = await asyncio.to_thread(store.snapshot, settings.owner_id)
            return {
                "enabled": True,
                "connected": await asyncio.to_thread(store.is_connected, settings.owner_id),
                "presence": {key: snapshot.get(key) for key in ("status", "busy_until", "refreshed_at")} if snapshot else None,
            }
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    @app.post("/v1/integrations/google-calendar/refresh")
    async def google_calendar_refresh(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        if not calendar_enabled():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Google Calendar integration is disabled")
        from gnaniassignment.google_calendar import CalendarIntegrationError, GoogleCalendarSettings, refresh_current_event
        try:
            settings = GoogleCalendarSettings.from_environment()
            store, client = calendar_services()
            snapshot = await asyncio.to_thread(refresh_current_event, store, client, settings.owner_id)
        except (CalendarIntegrationError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return {key: snapshot.get(key) for key in ("status", "busy_until")}

    @app.post("/v1/integrations/google-calendar/disconnect")
    async def google_calendar_disconnect(x_api_key: str | None = Header(default=None)) -> dict[str, bool]:
        _require_api_key(x_api_key)
        if not calendar_enabled():
            return {"disconnected": False}
        from gnaniassignment.google_calendar import GoogleCalendarSettings
        try:
            settings = GoogleCalendarSettings.from_environment()
            store, _ = calendar_services()
            await asyncio.to_thread(store.disconnect, settings.owner_id)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        await sessions.events.publish("calendar.disconnected")
        return {"disconnected": True}

    def work_actions():
        from gnaniassignment.work_items import PostgresWorkActionStore
        return PostgresWorkActionStore()

    @app.get("/v1/integrations/plane/health")
    async def plane_health(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        """Verify a server-side Plane token without returning it."""
        _require_api_key(x_api_key)
        from gnaniassignment.work_items import PlaneClient, WorkItemProviderError
        try:
            user = await asyncio.to_thread(PlaneClient().health)
        except (ValueError, WorkItemProviderError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return {"connected": True, "user_id": user.get("id"), "email": user.get("email")}

    @app.get("/v1/integrations/plane/work-items")
    async def plane_work_items(limit: int = 25, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        """Read only the small work-item projection needed by the app."""
        _require_api_key(x_api_key)
        from gnaniassignment.work_items import PlaneClient, WorkItemProviderError
        try:
            items = await asyncio.to_thread(PlaneClient().list_work_items, limit)
        except (ValueError, WorkItemProviderError) as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        return {"provider": "plane", "items": items}

    @app.post("/v1/work-actions/plane/comments", status_code=status.HTTP_201_CREATED)
    async def create_plane_comment_draft(payload: PlaneCommentDraftCreate, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        try:
            draft = await asyncio.to_thread(
                work_actions().create_comment_draft,
                owner_user_id=os.getenv("WORK_ACTION_OWNER_ID", "owner"),
                requested_by=payload.requested_by,
                project_id=payload.project_id,
                work_item_id=payload.work_item_id,
                comment=payload.comment,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        await sessions.events.publish("work_action.drafted", action_id=draft["id"], provider="plane", action_type="comment")
        return draft

    @app.get("/v1/work-actions")
    async def list_work_actions(status_value: str = Query(default="pending", alias="status"), limit: int = 100, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        try:
            actions = await asyncio.to_thread(work_actions().list_actions, os.getenv("WORK_ACTION_OWNER_ID", "owner"), status_value, limit)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
        return {"actions": actions}

    @app.post("/v1/work-actions/{action_id}/reject")
    async def reject_work_action(action_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(x_api_key)
        updated = await asyncio.to_thread(work_actions().reject, os.getenv("WORK_ACTION_OWNER_ID", "owner"), action_id)
        if not updated:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Work action is not pending or was not found")
        await sessions.events.publish("work_action.rejected", action_id=action_id)
        return {"id": action_id, "status": "rejected"}

    @app.post("/v1/work-actions/{action_id}/approve")
    async def approve_work_action(action_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        """Apply exactly one approved draft. The LLM never reaches this route."""
        _require_api_key(x_api_key)
        store = work_actions()
        claimed = await asyncio.to_thread(store.claim_for_approval, os.getenv("WORK_ACTION_OWNER_ID", "owner"), action_id)
        if claimed is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Work action is not pending or was already decided")
        try:
            if claimed["provider"] != "plane" or claimed["action_type"] != "comment":
                raise ValueError("Unsupported approved work action")
            draft = claimed["proposed_payload"]
            result = await asyncio.to_thread(
                PlaneClient().apply_comment,
                project_id=str(draft["project_id"]),
                work_item_id=str(draft["work_item_id"]),
                comment=str(draft["comment"]),
                external_id=f"voice-agent:{action_id}",
            )
        except (KeyError, ValueError, RuntimeError) as error:
            await asyncio.to_thread(store.mark_failed, action_id, str(error))
            await sessions.events.publish("work_action.failed", action_id=action_id, message=str(error)[:300])
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Plane did not accept the approved action") from error
        await asyncio.to_thread(store.mark_applied, action_id)
        await sessions.events.publish("work_action.applied", action_id=action_id, provider="plane", action_type="comment")
        return {"id": action_id, "status": "applied", "provider_result_id": result.get("id") if isinstance(result, dict) else None}

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

    @app.post("/v1/harness/conversations", status_code=status.HTTP_201_CREATED)
    async def harness_create_conversation(
        payload: HarnessConversationCreate,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Create a text-only conversation whose state stays isolated by ID."""
        _require_api_key(x_api_key)
        conversation_id = f"harness:{uuid4()}"
        session = await sessions.get_or_create(conversation_id, payload.customer_id)
        await sessions.events.publish(
            "harness.started", session_id=conversation_id, customer_id=payload.customer_id
        )
        return _harness_conversation_payload(conversation_id, session)

    @app.get("/v1/harness/conversations/{conversation_id}")
    async def harness_get_conversation(
        conversation_id: str,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_api_key(x_api_key)
        session = await sessions.get(conversation_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Harness conversation not found or already ended")
        return _harness_conversation_payload(conversation_id, session)

    @app.post("/v1/harness/conversations/{conversation_id}/turn")
    async def harness_turn(
        conversation_id: str,
        payload: HarnessConversationTurn,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Run one turn through the normal agent graph, without STT or TTS."""
        _require_api_key(x_api_key)
        session = await sessions.get(conversation_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Harness conversation not found or already ended")
        await sessions.events.publish(
            "harness.transcript.received", session_id=conversation_id, customer_id=session.customer_id, text=payload.text
        )
        result = await sessions.run_turn(conversation_id, payload.text)
        if result is None:  # Defensive: it may have ended while the request was queued.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Harness conversation not found or already ended")
        await sessions.events.publish(
            "harness.agent.responded", session_id=conversation_id, intent=result["intent"], text=result["spoken"]
        )
        state = _harness_conversation_payload(conversation_id, session)
        return {
            **state,
            "answer": result["spoken"],
            "intent": result["intent"],
            "topic": result.get("topic"),
            "metrics": result.get("metrics", {}),
        }

    @app.post("/v1/harness/conversations/{conversation_id}/end")
    async def harness_end_conversation(
        conversation_id: str,
        x_api_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_api_key(x_api_key)
        if await sessions.get(conversation_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Harness conversation not found or already ended")
        judgment = await sessions.end(conversation_id)
        await sessions.events.publish("harness.ended", session_id=conversation_id)
        return {"conversation_id": conversation_id, "ended": True, "judgment": judgment}

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
