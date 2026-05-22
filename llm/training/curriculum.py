"""Curriculum learning utilities.

Two practical curriculum strategies are supported out of the box:

* **Length curriculum** — start with short sequences (e.g. 2k) and gradually
  increase to ``max_seq_len`` over the first ``ramp_tokens`` tokens. Speeds
  early training by ~2-3× wall-clock with no measured quality loss.
* **Domain mixing curriculum** — linearly interpolate between an "easy" mix
  (Wikipedia, books) and the final mix (code, math, web) over training. Stage
  weights are specified as ``(start_weights, end_weights, ramp_tokens)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LengthCurriculum:
    start_seq_len: int = 2048
    end_seq_len: int = 8192
    ramp_tokens: int = 50_000_000_000  # 50B tokens
    schedule: str = "linear"  # or "exp"

    def length_for(self, tokens_seen: int) -> int:
        if tokens_seen >= self.ramp_tokens:
            return self.end_seq_len
        progress = tokens_seen / max(1, self.ramp_tokens)
        if self.schedule == "exp":
            progress = progress ** 2
        seq = int(self.start_seq_len + progress * (self.end_seq_len - self.start_seq_len))
        # Round to a multiple of 64 to keep kernel sizes friendly.
        return max(self.start_seq_len, (seq // 64) * 64)


@dataclass
class DomainMix:
    weights: dict[str, float]

    def normalize(self) -> DomainMix:
        total = sum(self.weights.values())
        return DomainMix(weights={k: v / total for k, v in self.weights.items()})


@dataclass
class MixingCurriculum:
    start: DomainMix
    end: DomainMix
    ramp_tokens: int = 200_000_000_000

    def mix_at(self, tokens_seen: int) -> DomainMix:
        progress = min(1.0, tokens_seen / max(1, self.ramp_tokens))
        keys = set(self.start.weights) | set(self.end.weights)
        return DomainMix(
            weights={
                k: (1 - progress) * self.start.weights.get(k, 0.0) + progress * self.end.weights.get(k, 0.0)
                for k in keys
            }
        ).normalize()


@dataclass
class CurriculumScheduler:
    """Top-level curriculum: combines length + domain mixing."""

    length: LengthCurriculum | None = None
    mixing: MixingCurriculum | None = None
    tokens_seen: int = 0
    history: list[dict] = field(default_factory=list)

    def step(self, batch_tokens: int) -> dict[str, object]:
        self.tokens_seen += batch_tokens
        state = {
            "tokens_seen": self.tokens_seen,
            "seq_len": self.length.length_for(self.tokens_seen) if self.length else None,
            "mix": self.mixing.mix_at(self.tokens_seen).weights if self.mixing else None,
        }
        self.history.append(state)
        return state
