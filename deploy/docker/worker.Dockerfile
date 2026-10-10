# GPU worker image: pipeline runner + Soup trainer in one image, so the runner
# launches the trainer entrypoint as a supervised subprocess on the same GPU host.
#   docker build -f deploy/docker/worker.Dockerfile -t mccp-worker .
# Pins below are the versions the recipe records; keep them in sync with
# examples/recipe-qwen3-8b-qlora-sft.yaml (engine + soup.version).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app

# Trainer environment (Soup + TRL/PEFT/bitsandbytes), isolated from the control plane.
RUN uv venv /opt/trainer --python python3.12 \
 && uv pip install --python /opt/trainer/bin/python \
      "soup-cli[train]==0.72.4" "trl==0.24.0" "peft==0.17.1" "transformers==4.57.1" \
      "bitsandbytes==0.48.1" "accelerate>=1.10" "huggingface_hub>=0.34" pyyaml mlflow-skinny
ENV PATH="/opt/trainer/bin:$PATH"

# Control-plane package (pipeline runner).
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /app/.venv --python python3.12 && uv sync --frozen --no-dev --no-install-project --python /app/.venv/bin/python
COPY src ./src
RUN uv sync --frozen --no-dev --python /app/.venv/bin/python
COPY trainer ./trainer
COPY policy ./policy
COPY deploy/fleet.yaml ./deploy/fleet.yaml

RUN useradd --uid 10001 --create-home mccp && mkdir -p /var/lib/mccp && chown -R mccp /var/lib/mccp
USER mccp
ENV MCCP_TRAINER_PYTHON=/opt/trainer/bin/python MCCP_TRAINER_ENTRY=/app/trainer/mccp_trainer_entry.py
CMD ["/app/.venv/bin/mccp", "worker"]
