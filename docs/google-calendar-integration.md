# Google Calendar presence integration

This integration is deliberately **presence-only** for voice calls. The server
refreshes the active event every 60 seconds and each call reads the cached
snapshot locally. The LLM receives only `available` / `unavailable until`;
event titles and other schedule data are not included in call context.

## Server configuration

After creating the Google OAuth web client, place these values only in the
server's ignored environment file or Secret Manager:

```dotenv
GOOGLE_CALENDAR_ENABLED=true
GOOGLE_CALENDAR_OWNER_ID=tushar
GOOGLE_CALENDAR_CLIENT_ID=<OAuth web-client ID>
GOOGLE_CALENDAR_CLIENT_SECRET=<OAuth web-client secret>
GOOGLE_CALENDAR_REDIRECT_URI=https://voice.sidecut.me/v1/integrations/google-calendar/callback
GOOGLE_CALENDAR_TOKEN_ENCRYPTION_KEY=<Fernet key>
GOOGLE_CALENDAR_STATE_SECRET=<random 32+ character secret>
GOOGLE_CALENDAR_REFRESH_SECONDS=60
```

Generate the encryption key locally; do not regenerate it after a calendar has
been connected because the key encrypts the stored refresh token:

```bash
uv run --extra google-calendar python -c \
  'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Generate the state secret with a password manager or:

```bash
openssl rand -hex 32
```

The OAuth client must have this exact Authorized redirect URI in Google Cloud:

```text
https://voice.sidecut.me/v1/integrations/google-calendar/callback
```

## Connect once

Use the authenticated dashboard/API to obtain the Google consent link:

```bash
curl -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/integrations/google-calendar/connect
```

Open the returned `authorization_url` in a browser and approve access. The
callback saves an encrypted refresh token in PostgreSQL and immediately writes
the first current-event snapshot.

Check safe status (it never returns an event title or token):

```bash
curl -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/integrations/google-calendar/status
```

Manual refresh is available for testing:

```bash
curl -X POST -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/integrations/google-calendar/refresh
```

To stop access and remove the server-side token/snapshot:

```bash
curl -X POST -H "X-API-Key: $AGENT_API_KEY" \
  https://voice.sidecut.me/v1/integrations/google-calendar/disconnect
```
