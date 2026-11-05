-- Versioned, tenant-scoped prompt trees. Apply after PostgreSQL is available.
CREATE TABLE IF NOT EXISTS prompt_trees (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, agent_id)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tree_id BIGINT NOT NULL REFERENCES prompt_trees(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at TIMESTAMPTZ,
    UNIQUE (tree_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS prompt_versions_one_active_idx
    ON prompt_versions(tree_id) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS prompt_nodes (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES prompt_versions(id) ON DELETE CASCADE,
    parent_node_id BIGINT REFERENCES prompt_nodes(id) ON DELETE CASCADE,
    condition JSONB NOT NULL DEFAULT '{}'::jsonb,
    instruction TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT true
);

CREATE INDEX IF NOT EXISTS prompt_nodes_version_priority_idx
    ON prompt_nodes(version_id, priority, id);
