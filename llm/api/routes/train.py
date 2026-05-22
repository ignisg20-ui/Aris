"""POST /train — submit a pretraining job.

For production multi-node training, the API just enqueues the job; the actual
launch is done by a Kubernetes controller using the manifests in ``deploy/k8s``.
This route runs the job *inside* the API process if ``ARIS_TRAINING_LOCAL=1``,
which is useful for development.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import TrainStartRequest, TrainStartResponse

router = APIRouter(prefix="", tags=["training"])

_JOBS: dict[str, dict[str, Any]] = {}


def _run_training(job_id: str, req: TrainStartRequest) -> None:
    _JOBS[job_id]["status"] = "running"
    try:
        from ...distributed.init import DistributedConfig
        from ...model.config import ArisConfig
        from ...training.data import StreamingShardedDataset, build_dataloader
        from ...training.trainer import Trainer, TrainingConfig

        cfg = ArisConfig.from_yaml(req.config_path)
        tcfg = TrainingConfig(
            total_steps=req.steps,
            checkpoint_dir=req.output_dir,
            distributed=DistributedConfig(),
        )
        from pathlib import Path

        shards = sorted(str(p) for p in Path(req.data_path).glob("*.bin"))
        if not shards:
            raise RuntimeError(f"no shards under {req.data_path}")
        ds = StreamingShardedDataset(shards=shards, seq_len=tcfg.micro_batch_size)
        loader = build_dataloader(ds, batch_size=tcfg.micro_batch_size)
        trainer = Trainer(cfg, tcfg, train_dataloader=loader)
        if req.resume_from:
            trainer.load_checkpoint(req.resume_from)
        trainer.fit()
        _JOBS[job_id]["status"] = "completed"
    except Exception as exc:
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["error"] = repr(exc)


@router.post("/train", response_model=TrainStartResponse)
async def start_training(body: TrainStartRequest, background: BackgroundTasks) -> TrainStartResponse:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "queued", "request": body.model_dump()}
    if os.environ.get("ARIS_TRAINING_LOCAL", "0") == "1":
        # Run in a background thread (single-process, single-GPU only).
        t = threading.Thread(target=_run_training, args=(job_id, body), daemon=True)
        t.start()
        return TrainStartResponse(job_id=job_id, status="running")
    # Production path: hand off to the controller. The implementation is platform-
    # specific (K8s Job / SLURM batch); we leave a hook here.
    raise HTTPException(
        status_code=501,
        detail="Submit jobs via the K8s controller. Set ARIS_TRAINING_LOCAL=1 for local dev.",
    )


@router.get("/train/{job_id}")
async def train_status(job_id: str) -> dict:
    if job_id not in _JOBS:
        raise HTTPException(404, "job not found")
    return _JOBS[job_id]
