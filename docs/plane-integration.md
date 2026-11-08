# Plane work-item integration

The agent never writes to Plane directly. A caller-proposed comment becomes a
row in `pending_work_actions`; only the authenticated dashboard owner can
approve it. Approval performs one tagged Plane API call. A duplicate retry is
guarded by the Plane comment `external_id` (`voice-agent:<action-id>`).

## 1. Create a Plane token

In Plane, open **Profile settings → Personal Access Tokens**, create a token
for the voice assistant, and copy it once. A workspace access token is the
better choice for a shared office integration when its permissions are
appropriate. Do not put either token in a browser, Flutter WebView, or Git.

The workspace slug is the segment in a Plane URL such as
`https://app.plane.so/acme-corp/`; its slug is `acme-corp`.

## 2. Configure the server only

Add the following values to the server's secret environment (for example GCP
Secret Manager plus the deployment environment):

```dotenv
WORK_ACTION_DATABASE_URL=postgresql://...
WORK_ACTION_OWNER_ID=tushar
PLANE_API_BASE_URL=https://api.plane.so
PLANE_WORKSPACE_SLUG=acme-corp
PLANE_API_KEY=plane_api_...
```

`WORK_ACTION_DATABASE_URL` may point at the existing agent PostgreSQL
database. The application creates `pending_work_actions` and its index on its
first use.

## 3. Verify the connection

```bash
curl -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/integrations/plane/health
```

A successful response contains `connected: true` and the Plane user ID. It
does not return the Plane token.

## 4. Inspect visible work items

```bash
curl -H "X-API-Key: $AGENT_API_KEY" \
  'https://voice.sidecut.me/v1/integrations/plane/work-items?limit=25'
```

The endpoint requests a deliberately small Plane v2 projection: identifier,
title, priority, state, target date, project, and assignee IDs.

## 5. Draft, then approve a comment

Creating a draft does not contact Plane:

```bash
curl -X POST -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_API_KEY" \
  -d '{
    "requested_by":"caller:+919999999999",
    "project_id":"<plane-project-uuid>",
    "work_item_id":"<plane-work-item-uuid>",
    "comment":"Testing is blocked pending staging credentials."
  }' \
  https://voice.sidecut.me/v1/work-actions/plane/comments
```

Review drafts:

```bash
curl -H "X-API-Key: $AGENT_API_KEY" \
  'https://voice.sidecut.me/v1/work-actions?status=pending'
```

Only this approval call posts the comment to Plane:

```bash
curl -X POST -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/work-actions/<action-id>/approve
```

Rejecting the draft makes no external call:

```bash
curl -X POST -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/work-actions/<action-id>/reject
```

## Current scope

This first slice supports read-only work-item retrieval and approved comments.
Assignment and state-change drafts should be added next using the same ledger.
They require an explicit provider-side read before applying because Plane
assignment updates replace the entire assignee set.
