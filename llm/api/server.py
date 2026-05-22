"""FastAPI server.

Exposes:

* ``GET  /health``     — liveness + model status.
* ``POST /generate``    — single-shot or streaming completion.
* ``POST /chat``        — chat-formatted completion (streaming optional).
* ``POST /train``       — kicks off a pretraining run on the local box.
* ``POST /finetune``    — SFT / RM / PPO / Constitutional fine-tuning job.
* ``GET  /metrics``     — Prometheus metrics endpoint.

For multi-host inference, run several instances behind a regional load-balancer
and use the K8s manifests in :mod:`deploy/k8s`.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..inference.engine import InferenceEngine
from ..model.aris import ArisForCausalLM
from ..model.config import ArisConfig
from ..monitoring.gpu import sample_gpu_stats
from ..monitoring.logging import configure_logging
from ..monitoring.metrics import MetricRegistry, metrics_app
from ..safety.refusal import RefusalPolicy
from ..tokenizer.tokenizer import ArisTokenizer
from .routes.chat import router as chat_router
from .routes.finetune import router as finetune_router
from .routes.generate import router as generate_router
from .routes.train import router as train_router
from .schemas import HealthResponse

logger = logging.getLogger("aris.api")


def _load_engine() -> InferenceEngine:
    config_path = os.environ.get("ARIS_MODEL_CONFIG", "llm/configs/1b.yaml")
    ckpt_path = os.environ.get("ARIS_CHECKPOINT")
    tokenizer_meta = os.environ.get("ARIS_TOKENIZER_META")

    config = ArisConfig.from_yaml(config_path)
    model = ArisForCausalLM(config)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        logger.info("Loaded checkpoint %s", ckpt_path)
    else:
        logger.warning("ARIS_CHECKPOINT not set — running with random weights")

    if tokenizer_meta and Path(tokenizer_meta).exists():
        tokenizer = ArisTokenizer.from_metadata(tokenizer_meta)
    else:
        # Sane default: tiktoken cl100k.
        tokenizer = ArisTokenizer(backend="tiktoken", model_path="cl100k_base.tiktoken")

    policy = RefusalPolicy()
    return InferenceEngine(model=model, tokenizer=tokenizer, safety_policy=policy)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=os.environ.get("ARIS_LOG_LEVEL", "INFO"))
    app.state.engine = _load_engine()
    app.state.metrics = MetricRegistry(namespace="aris")
    logger.info("Aris API started")
    try:
        yield
    finally:
        logger.info("Aris API shutting down")


def build_app() -> FastAPI:
    app = FastAPI(title="Aris LLM API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(generate_router)
    app.include_router(chat_router)
    app.include_router(train_router)
    app.include_router(finetune_router)
    app.mount("/metrics", metrics_app())

    # Bundled chat UI. Served as static files at /ui; root path redirects to it.
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="web-ui")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "engine", None)
        return HealthResponse(
            status="ok",
            model_loaded=engine is not None,
            gpu_available=torch.cuda.is_available(),
        )

    @app.get("/system")
    async def system() -> dict:
        return {
            "gpus": [g.__dict__ for g in sample_gpu_stats()],
            "cuda_available": torch.cuda.is_available(),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser("aris-serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("llm.api.server:build_app", host=args.host, port=args.port, factory=True, workers=args.workers)


if __name__ == "__main__":
    main()
