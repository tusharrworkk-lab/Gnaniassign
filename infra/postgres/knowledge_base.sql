CREATE TABLE IF NOT EXISTS approved_knowledge_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    intent TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'approved' CHECK (status IN ('draft', 'approved', 'retired')),
    priority SMALLINT NOT NULL DEFAULT 100,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS approved_knowledge_intent_idx
    ON approved_knowledge_chunks (intent, status, priority DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS approved_knowledge_search_idx
    ON approved_knowledge_chunks USING GIN (to_tsvector('simple', title || ' ' || content));
