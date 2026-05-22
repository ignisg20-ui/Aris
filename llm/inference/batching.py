"""Continuous (in-flight) batching.

Inspired by Orca / vLLM: instead of waiting for the whole batch to finish, each
decoding step picks up new requests whose prefills are ready and drops any
request that emitted ``EOS``. This is the single most important throughput
optimization in modern LLM serving.

This implementation is single-GPU and synchronous; for multi-GPU / async use the
``InferenceEngine`` directly behind a request-level queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import torch

from .engine import GenerationRequest, GenerationResult, InferenceEngine
from .sampling import sample_token


@dataclass
class RequestState:
    request: GenerationRequest
    input_ids: torch.Tensor
    generated: list[int] = field(default_factory=list)
    cache_offset: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished: bool = False
    finish_reason: str = ""
    text: str = ""


class ContinuousBatcher:
    """In-flight batched scheduler. Single-threaded event loop."""

    def __init__(self, engine: InferenceEngine, max_batch: int = 16, max_seq: int = 8192) -> None:
        self.engine = engine
        self.max_batch = max_batch
        self.max_seq = max_seq
        self._queue: Queue[RequestState] = Queue()
        self._results: dict[str, GenerationResult] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._stop = threading.Event()
        self._thread.start()

    def submit(self, request: GenerationRequest) -> str:
        ids = self.engine.tokenizer.encode(request.prompt)
        state = RequestState(
            request=request,
            input_ids=torch.tensor(ids, dtype=torch.long, device=self.engine.device),
        )
        self._queue.put(state)
        return request.request_id

    def get_result(self, request_id: str, timeout: float = 30.0) -> GenerationResult | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if request_id in self._results:
                    return self._results.pop(request_id)
            time.sleep(0.005)
        return None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        active: list[RequestState] = []
        while not self._stop.is_set():
            # Pull new requests up to capacity.
            while len(active) < self.max_batch:
                try:
                    state = self._queue.get_nowait()
                except Empty:
                    break
                active.append(state)
            if not active:
                time.sleep(0.001)
                continue

            self._step_batch(active)

            # Drain finished.
            for state in [s for s in active if s.finished]:
                with self._lock:
                    elapsed = (time.monotonic() - state.started_at) * 1000
                    self._results[state.request.request_id] = GenerationResult(
                        text=state.text,
                        prompt_tokens=int(state.input_ids.shape[0]),
                        completion_tokens=len(state.generated),
                        finish_reason=state.finish_reason or "stop",
                        latency_ms=elapsed,
                    )
                active.remove(state)

    @torch.inference_mode()
    def _step_batch(self, states: list[RequestState]) -> None:
        # Simplification: we process one decode step per loop iteration; prefill
        # is done up-front for new requests. A production implementation also
        # interleaves prefills via "chunked prefill".
        device = self.engine.device
        for state in states:
            if state.cache_offset == 0:
                input_ids = state.input_ids.unsqueeze(0)
                cache = self.engine._new_cache(1, self.max_seq)
                with torch.autocast(device_type=device.type, dtype=self.engine.dtype, enabled=device.type == "cuda"):
                    out = self.engine.model(input_ids, kv_cache=cache)
                state.cache_offset = input_ids.shape[1]
                state.__dict__["_cache"] = cache
                logits = out.logits[:, -1, :]
            else:
                step = torch.tensor([[state.generated[-1]]], dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=self.engine.dtype, enabled=device.type == "cuda"):
                    out = self.engine.model(step, kv_cache=state.__dict__["_cache"])
                state.cache_offset += 1
                logits = out.logits[:, -1, :]

            tok = sample_token(logits, state.request.sampling).item()
            state.generated.append(int(tok))
            if (
                tok == self.engine.tokenizer.eos_token_id
                or tok in state.request.sampling.stop_token_ids
                or len(state.generated) >= state.request.sampling.max_new_tokens
            ):
                state.finished = True
                state.finish_reason = "stop" if tok == self.engine.tokenizer.eos_token_id else "length"
                state.text = self.engine.tokenizer.decode(state.generated)
