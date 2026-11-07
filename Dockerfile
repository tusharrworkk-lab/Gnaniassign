# Production HTTP/webhook container. Voice capture/playback is handled by the
# telephony media adapter, not ALSA inside this cloud container.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libc++1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project \
    --extra api --extra bedrock --extra postgres-memory --extra redis-cache --extra gnani-stt --extra turn-detection --extra audio-enhancement

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable \
    --extra api --extra bedrock --extra postgres-memory --extra redis-cache --extra gnani-stt --extra turn-detection --extra audio-enhancement

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

CMD ["uvicorn", "gnaniassignment.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
