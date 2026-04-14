# ── Stage 1: Build ──
# We use a multi-stage build. The first stage installs dependencies.
# The second stage copies only what's needed to run. This keeps the
# final image small (~200MB instead of ~800MB).

FROM python:3.12-slim AS builder

# Install uv (our package manager) inside the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first. Docker caches each layer, so if
# pyproject.toml hasn't changed, it won't re-install dependencies.
# This makes rebuilds fast (seconds instead of minutes).
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies into a virtual environment
RUN uv sync --no-dev --no-install-project

# Now copy the rest of the source code
COPY src/ src/
COPY frontend/ frontend/

# Install our project itself
RUN uv sync --no-dev


# ── Stage 2: Runtime ──
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment and source from the builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/frontend /app/frontend
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Make sure the virtualenv's Python is used
ENV PATH="/app/.venv/bin:$PATH"
# Tell Python not to buffer output (so logs appear immediately)
ENV PYTHONUNBUFFERED=1

# Railway sets the PORT environment variable. Default to 8000 for local.
ENV PORT=8000

# Expose the port (documentation for humans, Docker doesn't enforce this)
EXPOSE ${PORT}

# Run the app. Key flags:
# --host 0.0.0.0: Listen on all interfaces (required in containers)
# --port $PORT: Use Railway's assigned port
# --workers 2: Two worker processes (good for a small instance)
# --proxy-headers + --forwarded-allow-ips="*": trust X-Forwarded-* from
#   Railway's ingress proxy so request.client.host is the real client IP
#   (slowapi rate limiting is per-IP; without this it sees the proxy IP
#   and becomes effectively global).
# We do NOT use --reload in production (that's for development only)
CMD uvicorn credrank.api:app --host 0.0.0.0 --port ${PORT} --workers 2 --proxy-headers --forwarded-allow-ips="*"
