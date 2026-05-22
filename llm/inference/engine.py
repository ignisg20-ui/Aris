"""High-level inference engine.

Provides a synchronous generate() API and an async streaming generator. Internally
the engine owns:
* The model in eval mode.
* A KV cache pool.
* The tokenizer & sampler.
* The safety policy (input + output filters).

For production serving the :class:`ContinuousBatcher` in :mod:`.batching` wraps
this engine for in-flight request batching.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

import torch

from ..model.aris import ArisForCausalLM
from ..model.config import ArisConfig
from ..model.kv_cache import KVCache
from ..safety.refusal import RefusalPolicy, evaluate_safety
from ..tokenizer.tokenizer import ArisTokenizer
from .sampling import SamplingParams, sample_token

logger = logging.getLogger("aris.infer")


@dataclass
class GenerationRequest:
    prompt: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = ""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length" | "refused"
    latency_ms: float


class InferenceEngine:
    def __init__(
        self,
        model: ArisForCausalLM,
        tokenizer: ArisTokenizer,
        device: torch.device | None = None,
        safety_policy: RefusalPolicy | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.safety = safety_policy
        self.config: ArisConfig = model.config
        self.dtype = dtype

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def _new_cache(self, batch_size: int, max_seq: int) -> KVCache:
        a = self.config.attention
        return KVCache.allocate(
            num_layers=self.num_layers,
            batch_size=batch_size,
            num_kv_heads=a.num_kv_heads,
            head_dim=a.head_dim or (self.config.hidden_size // a.num_heads),
            max_seq=max_seq,
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.monotonic()
        if self.safety is not None:
            verdict = evaluate_safety(self.safety, request.prompt)
            if not verdict.allowed:
                return GenerationResult(
                    text=verdict.refusal_text or "",
                    prompt_tokens=0,
                    completion_tokens=0,
                    finish_reason="refused",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

        ids = self.tokenizer.encode(request.prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        max_seq = input_ids.shape[1] + request.sampling.max_new_tokens
        cache = self._new_cache(batch_size=1, max_seq=max_seq)

        generated: list[int] = []
        generator: torch.Generator | None = None
        if request.sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.sampling.seed)

        # ----- Prefill -----
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = self.model(input_ids, kv_cache=cache)
        logits = out.logits[:, -1, :]
        next_token = sample_token(logits, request.sampling, generator=generator)
        generated.append(int(next_token.item()))

        finish_reason = "length"
        history = torch.tensor([generated], device=self.device)
        # ----- Decode loop -----
        for _ in range(request.sampling.max_new_tokens - 1):
            if generated[-1] in request.sampling.stop_token_ids or generated[-1] == self.tokenizer.eos_token_id:
                finish_reason = "stop"
                break
            input_step = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                out = self.model(input_step, kv_cache=cache)
            logits = out.logits[:, -1, :]
            next_token = sample_token(logits, request.sampling, token_history=history, generator=generator)
            tok = int(next_token.item())
            generated.append(tok)
            history = torch.cat([history, next_token.view(1, 1)], dim=-1)

        text = self.tokenizer.decode(generated)
        if self.safety is not None:
            verdict = evaluate_safety(self.safety, request.prompt, text)
            if not verdict.allowed:
                text = verdict.refusal_text or ""
                finish_reason = "refused"
        return GenerationResult(
            text=text,
            prompt_tokens=len(ids),
            completion_tokens=len(generated),
            finish_reason=finish_reason,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # ------------------------------------------------------------------ #
    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        """Synchronous streaming generator yielding text deltas."""
        if self.safety is not None:
            verdict = evaluate_safety(self.safety, request.prompt)
            if not verdict.allowed:
                yield verdict.refusal_text or ""
                return

        ids = self.tokenizer.encode(request.prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        cache = self._new_cache(1, input_ids.shape[1] + request.sampling.max_new_tokens)
        decoder = self.tokenizer.stream_decoder()
        history_tokens: list[int] = []

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                out = self.model(input_ids, kv_cache=cache)
            logits = out.logits[:, -1, :]
            next_token = sample_token(logits, request.sampling)
            history_tokens.append(int(next_token.item()))
            yield decoder.step(history_tokens[-1])

            for _ in range(request.sampling.max_new_tokens - 1):
                if history_tokens[-1] == self.tokenizer.eos_token_id or history_tokens[-1] in request.sampling.stop_token_ids:
                    break
                step = torch.tensor([[history_tokens[-1]]], dtype=torch.long, device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                    out = self.model(step, kv_cache=cache)
                logits = out.logits[:, -1, :]
                next_token = sample_token(
                    logits, request.sampling, token_history=torch.tensor([history_tokens], device=self.device)
                )
                history_tokens.append(int(next_token.item()))
                delta = decoder.step(history_tokens[-1])
                if delta:
                    yield delta

        tail = decoder.flush()
        if tail:
            yield tail

    async def generate_stream_async(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Asynchronous wrapper for use in FastAPI streaming endpoints."""
        import asyncio

        loop = asyncio.get_running_loop()
        # We run the synchronous generator on a thread to avoid blocking the loop.
        gen = self.generate_stream(request)
        while True:
            chunk = await loop.run_in_executor(None, lambda: next(gen, None))
            if chunk is None:
                break
            yield chunk
