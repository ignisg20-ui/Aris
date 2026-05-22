"""Streaming helpers: Server-Sent Events wrapper."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class StreamingResponse:
    """SSE-style frames for ``/chat`` and ``/generate`` streaming endpoints."""

    @staticmethod
    def text_chunk(text: str) -> dict:
        return {"type": "delta", "text": text}

    @staticmethod
    def end(finish_reason: str) -> dict:
        return {"type": "end", "finish_reason": finish_reason}

    @staticmethod
    def error(message: str) -> dict:
        return {"type": "error", "message": message}

    @staticmethod
    async def encode(stream: AsyncIterator[dict]) -> AsyncIterator[str]:
        async for event in stream:
            yield "data: " + json.dumps(event) + "\n\n"
