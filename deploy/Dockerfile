# Multi-stage Dockerfile for Aris.
#
# Build:
#   docker build -t aris:inference -f deploy/Dockerfile --target inference .
#   docker build -t aris:train     -f deploy/Dockerfile --target train     .

# -------- Base --------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl ca-certificates \
        build-essential libnuma1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /opt/aris

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip wheel && pip install -r requirements.txt

COPY . .
RUN pip install -e .

# -------- Inference stage --------
FROM base AS inference
ENV ARIS_LOG_LEVEL=INFO
EXPOSE 8000
CMD ["python", "-m", "llm.api.server", "--host", "0.0.0.0", "--port", "8000"]

# -------- Training stage --------
FROM base AS train
RUN pip install -e ".[train]"
CMD ["python", "-m", "llm.training.pretrain", "--help"]
