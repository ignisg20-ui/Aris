"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..inference.sampling import SamplingParams


class GenerateRequest(BaseModel):
    prompt: str
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    stream: bool = False
    request_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: float


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    stream: bool = False
    request_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class ChatCompletionResponse(BaseModel):
    message: ChatMessage
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_ms: float


class TrainStartRequest(BaseModel):
    config_path: str
    data_path: str
    output_dir: str
    steps: int = 1000
    resume_from: str | None = None


class TrainStartResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "failed", "completed"] = "queued"


class FinetuneRequest(BaseModel):
    base_checkpoint: str
    dataset_path: str
    output_dir: str
    method: Literal["sft", "rm", "ppo", "constitutional"] = "sft"
    epochs: int = 3
    learning_rate: float = 2e-5


class FinetuneResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "failed", "completed"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    gpu_available: bool
