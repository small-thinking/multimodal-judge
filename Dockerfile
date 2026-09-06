# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.11-slim-bookworm AS base
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" HF_HOME=/app/cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

FROM base AS cpu
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra cpu --extra vlm
ENTRYPOINT ["multimodal-judge"]
CMD ["smoke", "--config", "configs/cpu.yaml"]

# CUDA wheels include user-space libraries. Host needs an NVIDIA driver + Container Toolkit.
FROM base AS cuda
RUN test "$(uname -m)" = "x86_64"
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra cuda --extra vlm
ENTRYPOINT ["multimodal-judge"]
CMD ["smoke", "--config", "configs/gpu.yaml"]

# A plain `docker build .` defaults to the portable CPU image.
FROM cpu AS default
