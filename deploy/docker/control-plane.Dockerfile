# Control-plane image (API and CPU-lane workers; no GPU). Build from repo root:
#   docker build -f deploy/docker/control-plane.Dockerfile -t mccp-api .
# uv is pinned to the version that produced uv.lock (`uv --version` when relocking).
ARG UV_VERSION=0.11.32
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
# Directories that named volumes mount over must exist and be owned by the
# runtime user: Docker copies this ownership into a fresh named volume.
RUN useradd --uid 10001 --create-home mccp \
 && mkdir -p /var/lib/mccp/objects /var/lib/mccp/work /var/lib/mccp/lora \
 && chown -R mccp:mccp /var/lib/mccp
WORKDIR /app
COPY --from=build /app /app
COPY policy ./policy
COPY deploy/fleet.yaml ./deploy/fleet.yaml
COPY profiles ./profiles
COPY deploy/sql ./deploy/sql
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER mccp
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["mccp", "api", "--host", "0.0.0.0", "--port", "8080"]
