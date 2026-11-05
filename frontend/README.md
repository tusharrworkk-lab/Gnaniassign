# Relay mobile dashboard

React/Vite frontend intentionally authored for a 390 × 844 mobile viewport and later Flutter WebView use.

```bash
cd frontend
npm install
npm run dev
```

To connect it to the local API, create `frontend/.env.local` (it is ignored by
Git):

```dotenv
VITE_AGENT_API_URL=http://127.0.0.1:8080
VITE_AGENT_API_KEY=the-same-value-as-AGENT_API_KEY
```

Then start the API with PostgreSQL and Redis available. Apply the prompt-tree
schema before the first run:

```bash
psql "$PROMPT_TREE_DATABASE_URL" -f infra/postgres/prompt_trees.sql
psql "$PROMPT_TREE_DATABASE_URL" -f infra/postgres/personal_phone_prompt_tree.sql
uv run uvicorn gnaniassignment.api.server:app --host 0.0.0.0 --port 8080
```

The **Setup** tab publishes each added instruction as an immutable prompt-tree
version; existing calls keep their already-resolved prompt and new calls use
the new version. The **Calls** tab consumes the API's live event stream for
STT, response, and MiniLM shadow events.

For shadow classification, keep the live router generic and enable the
background model only after its local checkpoint and `retail-classifier` extra
are installed:

```dotenv
INTENT_PROVIDER=generic
SHADOW_INTENT_ENABLED=true
RETAIL_MINILM_MODEL=/absolute/path/to/systemone-retail-v1
```

`VITE_AGENT_API_KEY` is exposed to the browser bundle. It is acceptable for a
local workstation only; deploy the WebView behind an authenticated, same-origin
backend session rather than embedding a production API key.
