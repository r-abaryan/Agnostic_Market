# syntax=docker/dockerfile:1.7
# UNVERIFIED: this image has never been built. Written and reviewed by inspection only,
# because no Docker daemon was available. Dependency resolution, the livekit-agents native
# rtc bindings, and any further system libraries are unconfirmed. Do not treat a passing
# review of this file as evidence that it builds or that the worker starts.
# Immutable network-worker artifact. Its pushed manifest digest is the value that
# VOICE_AGENT_BUILD_ARTIFACT_DIGEST must carry into certification and activation.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lock alone, so this layer survives source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# TLS roots are required for LiveKit, the model providers, and PostgreSQL.
RUN apt-get update \
    && apt-get install --no-install-recommends --assume-yes ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

RUN useradd --system --create-home --uid 10001 agent

WORKDIR /app

COPY --from=builder --chown=agent:agent /app/.venv ./.venv
COPY --chown=agent:agent src ./src
COPY --chown=agent:agent scripts ./scripts
# voice_agent.py resolves its registry at ../config relative to itself.
COPY --chown=agent:agent config ./config
COPY --chown=agent:agent assets/audio ./assets/audio

# voice_agent.py loads both of these before it accepts any job. They are generated
# evidence rather than source, so fail the build instead of publishing a worker that
# cannot start.
RUN test -f config/conformance/reports.json || (echo "missing config/conformance/reports.json" >&2; exit 1)
RUN test -f config/telemetry/semantic_routing_report.json || (echo "missing config/telemetry/semantic_routing_report.json" >&2; exit 1)

USER agent

ENTRYPOINT ["python", "scripts/voice_agent.py"]
CMD ["start"]
