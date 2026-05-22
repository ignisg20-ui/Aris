"""POST /generate — single-shot or streaming text completion."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...inference.engine import GenerationRequest
from ..schemas import GenerateRequest, GenerateResponse

router = APIRouter(prefix="", tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest):
    engine = request.app.state.engine
    metrics = request.app.state.metrics
    if engine is None:
        raise HTTPException(503, "engine not ready")

    gen_req = GenerationRequest(
        prompt=body.prompt,
        sampling=body.sampling,
        request_id=body.request_id or str(uuid.uuid4()),
    )

    if body.stream:
        async def event_stream():
            async for chunk in engine.generate_stream_async(gen_req):
                yield "data: " + json.dumps({"type": "delta", "text": chunk}) + "\n\n"
            yield "data: " + json.dumps({"type": "end"}) + "\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = engine.generate(gen_req)
    metrics.counter("requests_total", labelnames=("endpoint",)).labels(endpoint="generate").inc()
    metrics.histogram("latency_seconds", labelnames=("endpoint",)).labels(endpoint="generate").observe(result.latency_ms / 1000)
    metrics.counter("tokens_generated_total").inc(result.completion_tokens)
    return GenerateResponse(**result.__dict__)
