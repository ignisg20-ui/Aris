"""POST /chat — chat-formatted completion."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...inference.engine import GenerationRequest
from ...tokenizer.tokenizer import ChatMessage as TokChatMessage
from ..schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage

router = APIRouter(prefix="", tags=["chat"])


@router.post("/chat", response_model=ChatCompletionResponse)
async def chat(request: Request, body: ChatCompletionRequest):
    engine = request.app.state.engine
    metrics = request.app.state.metrics
    if engine is None:
        raise HTTPException(503, "engine not ready")

    tokenizer_messages = [TokChatMessage(role=m.role, content=m.content, name=m.name) for m in body.messages]
    prompt = engine.tokenizer.chat_template.render(tokenizer_messages, add_generation_prompt=True)

    gen_req = GenerationRequest(
        prompt=prompt,
        sampling=body.sampling,
        request_id=body.request_id or str(uuid.uuid4()),
    )

    if body.stream:
        async def event_stream():
            try:
                async for chunk in engine.generate_stream_async(gen_req):
                    yield "data: " + json.dumps({"type": "delta", "text": chunk}) + "\n\n"
                yield "data: " + json.dumps({"type": "end"}) + "\n\n"
            except Exception as exc:  # noqa: BLE001 — surfaced to client over SSE
                yield "data: " + json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    result = engine.generate(gen_req)
    metrics.counter("requests_total", labelnames=("endpoint",)).labels(endpoint="chat").inc()
    metrics.histogram("latency_seconds", labelnames=("endpoint",)).labels(endpoint="chat").observe(result.latency_ms / 1000)
    metrics.counter("tokens_generated_total").inc(result.completion_tokens)

    return ChatCompletionResponse(
        message=ChatMessage(role="assistant", content=result.text),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
    )
