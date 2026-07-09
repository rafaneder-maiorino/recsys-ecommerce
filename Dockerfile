# syntax=docker/dockerfile:1

# ---------- Stage 1: builder — resolves and installs dependencies ----------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependency layer first: rebuilds only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer: cheap to rebuild on code changes.
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# ---------- Stage 2: runtime — slim image with a non-root user ----------
FROM python:3.11-slim AS runtime

# git is required by DVC to resolve the repository state.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 mluser \
    && git config --system safe.directory '*'

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER mluser

# The project workspace is bind-mounted by docker-compose at runtime,
# so data, DVC state and Git history stay on the host.
CMD ["dvc", "repro"]
