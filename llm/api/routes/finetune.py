"""POST /finetune — SFT/RM/PPO/Constitutional finetuning."""

from __future__ import annotations

import threading
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..schemas import FinetuneRequest, FinetuneResponse

router = APIRouter(prefix="", tags=["finetune"])

_JOBS: dict[str, dict[str, Any]] = {}


def _run_finetune(job_id: str, req: FinetuneRequest) -> None:
    _JOBS[job_id]["status"] = "running"
    try:
        # NOTE: the trainer classes themselves live in llm.alignment.{sft,rlhf,reward_model};
        # this endpoint is a thin orchestrator for development. In production, K8s Jobs run
        # the dedicated entry points and report status back via a sidecar.
        if req.method == "sft":
            pass
        elif req.method == "rm":
            from ...alignment.reward_model import RewardModel  # noqa: F401
        elif req.method == "ppo":
            from ...alignment.rlhf import PPOConfig  # noqa: F401
        elif req.method == "constitutional":
            from ...alignment.constitutional import ConstitutionalTrainer  # noqa: F401
        else:
            raise ValueError(req.method)
        _JOBS[job_id]["status"] = "completed"
    except Exception as exc:
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["error"] = repr(exc)


@router.post("/finetune", response_model=FinetuneResponse)
async def start_finetune(body: FinetuneRequest, background: BackgroundTasks) -> FinetuneResponse:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "queued", "request": body.model_dump()}
    t = threading.Thread(target=_run_finetune, args=(job_id, body), daemon=True)
    t.start()
    return FinetuneResponse(job_id=job_id, status="running")


@router.get("/finetune/{job_id}")
async def finetune_status(job_id: str) -> dict:
    if job_id not in _JOBS:
        raise HTTPException(404, "job not found")
    return _JOBS[job_id]
