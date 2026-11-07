"""Fast, versioned prompt-tree selection.

PostgreSQL is the source of truth. A process-local cache is the normal hot
path; Redis is an optional shared cache for multi-process deployments.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PromptNode:
    node_id: str
    parent_id: str | None
    condition: dict[str, Any]
    instruction: str
    priority: int


@dataclass(frozen=True)
class PromptPlan:
    tenant_id: str
    agent_id: str
    version: int
    nodes: tuple[PromptNode, ...]

    def render(self, context: dict[str, Any]) -> str:
        """Traverse matching nodes deterministically; input text never routes it."""
        selected: set[str] = set()
        fragments: list[str] = []
        for node in sorted(self.nodes, key=lambda value: (value.priority, value.node_id)):
            if node.parent_id is not None and node.parent_id not in selected:
                continue
            if _matches(node.condition, context):
                selected.add(node.node_id)
                if node.instruction.strip():
                    fragments.append(node.instruction.strip())
        return "\n\n".join(fragments)

    def to_json(self) -> str:
        return json.dumps({
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "version": self.version,
            "nodes": [node.__dict__ for node in self.nodes],
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "PromptPlan":
        value = json.loads(raw)
        return cls(
            tenant_id=value["tenant_id"], agent_id=value["agent_id"], version=int(value["version"]),
            nodes=tuple(PromptNode(**node) for node in value["nodes"]),
        )


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    """Small allowlisted condition language: exact value or one-of list."""
    for key, expected in condition.items():
        actual = context.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class PromptTreeStore(Protocol):
    def load_active(self, tenant_id: str, agent_id: str) -> PromptPlan | None: ...


class PromptPlanCache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...


class RedisPromptPlanCache:
    def __init__(self, url: str) -> None:
        try:
            from redis import Redis
        except ImportError as error:
            raise RuntimeError("Install Redis support with: uv sync --extra redis-cache") from error
        self.client = Redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self.client.delete(key)


class PostgresPromptTreeStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("PROMPT_TREE_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set PROMPT_TREE_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")

    def load_active(self, tenant_id: str, agent_id: str) -> PromptPlan | None:
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL support with: uv sync --extra postgres-memory") from error
        with psycopg.connect(self.database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT version.id, version.version
                FROM prompt_trees tree
                JOIN prompt_versions version ON version.tree_id = tree.id
                WHERE tree.tenant_id = %s AND tree.agent_id = %s AND version.status = 'active'
                ORDER BY version.version DESC LIMIT 1
                """,
                (tenant_id, agent_id),
            )
            active = cursor.fetchone()
            if active is None:
                return None
            version_id, version = active
            cursor.execute(
                """
                SELECT id::text, parent_node_id::text, condition, instruction, priority
                FROM prompt_nodes
                WHERE version_id = %s AND enabled = true
                ORDER BY priority, id
                """,
                (version_id,),
            )
            nodes = tuple(
                PromptNode(node_id=row[0], parent_id=row[1], condition=row[2] or {}, instruction=row[3], priority=row[4])
                for row in cursor.fetchall()
            )
        return PromptPlan(tenant_id=tenant_id, agent_id=agent_id, version=version, nodes=nodes)


class PostgresPromptTreeEditor:
    """Publish a new immutable prompt-plan version for dashboard edits.

    The dashboard never mutates a version currently used by a call.  It clones
    the active tree, appends the requested node, then atomically promotes the
    clone.  Existing in-flight sessions keep their already-resolved prompt.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("PROMPT_TREE_DATABASE_URL") or os.getenv("EPISODIC_MEMORY_DATABASE_URL")
        if not self.database_url:
            raise ValueError("Set PROMPT_TREE_DATABASE_URL or EPISODIC_MEMORY_DATABASE_URL")

    def append_node(
        self,
        tenant_id: str,
        agent_id: str,
        instruction: str,
        priority: int = 100,
        condition: dict[str, Any] | None = None,
    ) -> PromptPlan:
        instruction = instruction.strip()
        if not 1 <= len(instruction) <= 4000:
            raise ValueError("instruction must contain between 1 and 4000 characters")
        if not -10_000 <= priority <= 10_000:
            raise ValueError("priority must be between -10000 and 10000")
        condition = condition or {}
        _validate_condition(condition)
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Install PostgreSQL support with: uv sync --extra postgres-memory") from error

        with psycopg.connect(self.database_url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM prompt_trees WHERE tenant_id = %s AND agent_id = %s FOR UPDATE",
                (tenant_id, agent_id),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO prompt_trees (tenant_id, agent_id) VALUES (%s, %s) RETURNING id",
                    (tenant_id, agent_id),
                )
                tree_id = cursor.fetchone()[0]
            else:
                tree_id = row[0]

            cursor.execute(
                "SELECT id, version FROM prompt_versions WHERE tree_id = %s AND status = 'active' FOR UPDATE",
                (tree_id,),
            )
            active = cursor.fetchone()
            old_nodes: list[tuple[Any, ...]] = []
            if active is not None:
                old_version_id, old_version = active
                cursor.execute(
                    "SELECT id, parent_node_id, condition, instruction, priority, enabled FROM prompt_nodes WHERE version_id = %s ORDER BY priority, id",
                    (old_version_id,),
                )
                old_nodes = list(cursor.fetchall())
                cursor.execute("UPDATE prompt_versions SET status = 'archived' WHERE id = %s", (old_version_id,))
                next_version = old_version + 1
            else:
                cursor.execute("SELECT COALESCE(MAX(version), 0) FROM prompt_versions WHERE tree_id = %s", (tree_id,))
                next_version = cursor.fetchone()[0] + 1

            cursor.execute(
                "INSERT INTO prompt_versions (tree_id, version, status, activated_at) VALUES (%s, %s, 'active', now()) RETURNING id",
                (tree_id, next_version),
            )
            version_id = cursor.fetchone()[0]
            copied_ids: dict[int, int] = {}
            copied_parents: list[tuple[int, int | None]] = []
            for old_id, old_parent, old_condition, old_instruction, old_priority, enabled in old_nodes:
                cursor.execute(
                    "INSERT INTO prompt_nodes (version_id, parent_node_id, condition, instruction, priority, enabled) VALUES (%s, NULL, %s::jsonb, %s, %s, %s) RETURNING id",
                    (version_id, json.dumps(old_condition or {}), old_instruction, old_priority, enabled),
                )
                copied_ids[old_id] = cursor.fetchone()[0]
                copied_parents.append((old_id, old_parent))
            for old_id, old_parent in copied_parents:
                if old_parent is not None and old_parent in copied_ids:
                    cursor.execute("UPDATE prompt_nodes SET parent_node_id = %s WHERE id = %s", (copied_ids[old_parent], copied_ids[old_id]))
            cursor.execute(
                "INSERT INTO prompt_nodes (version_id, condition, instruction, priority) VALUES (%s, %s::jsonb, %s, %s)",
                (version_id, json.dumps(condition), instruction, priority),
            )

        plan = PostgresPromptTreeStore(self.database_url).load_active(tenant_id, agent_id)
        if plan is None:  # Defensive: a committed active plan must be visible.
            raise RuntimeError("Prompt version was published but could not be reloaded")
        return plan


def _validate_condition(condition: dict[str, Any]) -> None:
    if not isinstance(condition, dict):
        raise ValueError("condition must be an object")
    # customer_id is intentionally an exact-match condition.  It allows the
    # dashboard to publish an instruction for one opted-in phone owner without
    # accidentally exposing it to every other caller.
    allowed = {"agent", "channel", "language", "customer_id"}
    unknown = set(condition) - allowed
    if unknown:
        raise ValueError(f"condition contains unsupported keys: {', '.join(sorted(unknown))}")
    for key, value in condition.items():
        if isinstance(value, str):
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        raise ValueError(f"condition.{key} must be a string or a list of strings")


class PromptPlanResolver:
    """L1-memory → Redis → PostgreSQL prompt-plan resolution."""

    _l1: dict[str, tuple[float, PromptPlan]] = {}

    def __init__(self, store: PromptTreeStore, cache: PromptPlanCache | None = None, ttl_seconds: int = 300) -> None:
        self.store, self.cache, self.ttl_seconds = store, cache, ttl_seconds

    @staticmethod
    def cache_key(tenant_id: str, agent_id: str) -> str:
        return f"prompt:active:{tenant_id}:{agent_id}"

    def resolve(self, tenant_id: str, agent_id: str) -> PromptPlan | None:
        key = self.cache_key(tenant_id, agent_id)
        cached = self._l1.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        if self.cache is not None:
            try:
                raw = self.cache.get(key)
                if raw:
                    plan = PromptPlan.from_json(raw)
                    self._remember(key, plan)
                    return plan
            except Exception:
                # Redis is an optimization; never make a call fail because a
                # shared cache is temporarily unavailable.
                pass
        plan = self.store.load_active(tenant_id, agent_id)
        if plan is None:
            return None
        self._remember(key, plan)
        if self.cache is not None:
            try:
                self.cache.set(key, plan.to_json(), self.ttl_seconds)
            except Exception:
                pass
        return plan

    def invalidate(self, tenant_id: str, agent_id: str) -> None:
        key = self.cache_key(tenant_id, agent_id)
        self._l1.pop(key, None)
        if self.cache is not None:
            try:
                self.cache.delete(key)
            except Exception:
                pass

    def _remember(self, key: str, plan: PromptPlan) -> None:
        self._l1[key] = (time.monotonic() + self.ttl_seconds, plan)
