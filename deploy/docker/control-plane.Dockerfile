# Control-plane API image (no GPU). Build from repo root:
#   docker build -f deploy/docker/control-plane.Dockerfile -t mccp-api .
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.13-slim
RUN useradd --uid 10001 --create-home mccp && mkdir -p /var/lib/mccp/objects /var/lib/mccp/work && chown -R mccp /var/lib/mccp
WORKDIR /app
COPY --from=build /app /app
COPY policy ./policy
COPY deploy/fleet.yaml ./deploy/fleet.yaml
COPY deploy/sql ./deploy/sql
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER mccp
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["mccp", "api", "--host", "0.0.0.0", "--port", "8080"]
