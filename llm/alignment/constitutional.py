"""Constitutional AI (Bai et al. 2022).

Two-phase pipeline:

1. **Constitutional revision (SL-CAI)** — for each (prompt, initial response),
   sample a *critique* and a *revision* using a set of constitutional principles
   (free-form natural-language rules). Train an SFT model on the *revised*
   responses, giving a self-improved supervised dataset that pushes the model
   toward following the constitution.

2. **RLAIF (RL-CAI)** — replace the human preference label in RLHF with an
   AI-generated preference from a stronger reference model conditioned on the
   constitution.

Math: identical to RLHF (Bradley-Terry → PPO) but with AI-labelled preferences.
The constitution acts as the prompt for the labeller, so it directly shapes
``r_φ(x, y)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger("aris.alignment.constitutional")


@dataclass
class ConstitutionalPrinciple:
    name: str
    critique_request: str
    revision_request: str

    def render_critique_prompt(self, prompt: str, response: str) -> str:
        return (
            f"Human: {prompt}\n\nAssistant: {response}\n\n"
            f"Critique request: {self.critique_request}\n"
            "Critique:"
        )

    def render_revision_prompt(self, prompt: str, response: str, critique: str) -> str:
        return (
            f"Human: {prompt}\n\nAssistant: {response}\n\n"
            f"Critique: {critique}\n\n"
            f"Revision request: {self.revision_request}\n"
            "Revised response:"
        )


DEFAULT_PRINCIPLES: tuple[ConstitutionalPrinciple, ...] = (
    ConstitutionalPrinciple(
        name="harmfulness",
        critique_request="Identify specific ways in which the assistant's response is harmful, unethical, or socially biased.",
        revision_request="Rewrite the assistant's response to remove harmful, unethical, or biased content while preserving helpfulness.",
    ),
    ConstitutionalPrinciple(
        name="honesty",
        critique_request="Identify any false, misleading, or speculative claims and any failures to express uncertainty.",
        revision_request="Rewrite the response to be honest, accurate, and to clearly express uncertainty where appropriate.",
    ),
    ConstitutionalPrinciple(
        name="helpfulness",
        critique_request="Identify any way in which the response is unhelpful, vague, or evasive without good reason.",
        revision_request="Rewrite the response so it directly and constructively addresses the user's actual question.",
    ),
)


@dataclass
class ConstitutionalRevision:
    prompt: str
    original_response: str
    critique: str
    revised_response: str
    principle_name: str


@dataclass
class ConstitutionalTrainer:
    """Generates self-critique → revision pairs from a base model."""

    generate_fn: Callable[[str], str]  # callable that takes prompt and returns model response
    principles: Sequence[ConstitutionalPrinciple] = field(default_factory=lambda: DEFAULT_PRINCIPLES)

    def revise(self, prompt: str, response: str) -> ConstitutionalRevision:
        # Apply principles sequentially; each revision is fed into the next critique.
        current = response
        last_revision: ConstitutionalRevision | None = None
        for principle in self.principles:
            critique = self.generate_fn(principle.render_critique_prompt(prompt, current)).strip()
            revised = self.generate_fn(principle.render_revision_prompt(prompt, current, critique)).strip()
            last_revision = ConstitutionalRevision(
                prompt=prompt,
                original_response=response,
                critique=critique,
                revised_response=revised,
                principle_name=principle.name,
            )
            current = revised
        assert last_revision is not None
        return last_revision

    def generate_dataset(self, prompts: Iterable[str]) -> list[ConstitutionalRevision]:
        out: list[ConstitutionalRevision] = []
        for i, prompt in enumerate(prompts):
            response = self.generate_fn(prompt)
            rev = self.revise(prompt, response)
            out.append(rev)
            if (i + 1) % 100 == 0:
                logger.info("Generated %d revisions", i + 1)
        return out


def ai_preference_label(
    judge_fn: Callable[[str], str],
    constitution: str,
    prompt: str,
    response_a: str,
    response_b: str,
) -> int:
    """Ask an AI judge to pick the better response. Returns 0 (A) or 1 (B).

    The judge prompt is the same Bai et al. used for RL-CAI labelling.
    """
    judge_prompt = (
        f"Constitution:\n{constitution}\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Following the constitution above, which response is more helpful, honest and harmless?\n"
        "Answer with just `A` or `B`:"
    )
    answer = judge_fn(judge_prompt).strip().upper()
    return 0 if answer.startswith("A") else 1
