# GPU worker image: pipeline runner + trainer environment in one image, so the
# runner launches the trainer entrypoint as a supervised subprocess.
#   docker build -f deploy/docker/worker.Dockerfile -t mccp-worker .
# Trainer dependencies are hash-locked in trainer/requirements.lock.txt
# (regenerate with the command at the top of that file). Keep the recipe's
# `engine` and `soup.version` fields equal to those pins.
ARG UV_VERSION=0.11.32
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
# Ubuntu 24.04 ships Python 3.12, which both the trainer and control plane target.
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

# Trainer environment (Soup + TRL/PEFT/bitsandbytes), isolated from the control plane.
COPY trainer/requirements.lock.txt /tmp/trainer-requirements.lock.txt
RUN uv venv /opt/trainer --python /usr/bin/python3.12 \
 && uv pip install --python /opt/trainer/bin/python --require-hashes -r /tmp/trainer-requirements.lock.txt

# Control-plane package (pipeline runner).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --python /usr/bin/python3.12
COPY src ./src
RUN uv sync --frozen --no-dev --python /usr/bin/python3.12
COPY trainer ./trainer
COPY policy ./policy
COPY deploy/fleet.yaml ./deploy/fleet.yaml
COPY profiles ./profiles

# Volume mount points owned by the runtime user (copied into fresh named volumes).
RUN useradd --uid 10001 --create-home mccp \
 && mkdir -p /var/lib/mccp/objects /var/lib/mccp/work /var/lib/mccp/lora /home/mccp/.cache/huggingface \
 && chown -R mccp:mccp /var/lib/mccp /home/mccp/.cache
USER mccp
ENV PATH="/app/.venv/bin:/opt/trainer/bin:$PATH" \
    MCCP_TRAINER_PYTHON=/opt/trainer/bin/python \
    MCCP_TRAINER_ENTRY=/app/trainer/mccp_trainer_entry.py \
    HF_HOME=/home/mccp/.cache/huggingface
CMD ["/app/.venv/bin/mccp", "worker"]
