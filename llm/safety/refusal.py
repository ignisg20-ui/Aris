"""Refusal policies.

A refusal policy is a deterministic, rule-based filter applied to both *input*
prompts and *output* generations. It is meant to complement (not replace) a
learned harmlessness classifier and the model's own RLHF refusal behavior.

We intentionally keep this module dependency-free and synchronous so it can run
in front of every request with negligible latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class RefusalReason(str, Enum):
    SAFE = "safe"
    DISALLOWED_TOPIC = "disallowed_topic"
    POLICY_VIOLATION = "policy_violation"
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"


_DISALLOWED_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:bio|chemical|nuclear)\s*weapon", re.I),
    re.compile(r"\b(?:csam|child\s+sexual)\b", re.I),
    re.compile(r"\bhow\s+to\s+(?:make|build|synthesize)\s+(?:meth|fentanyl|napalm|nerve\s+agent)", re.I),
)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (?:all|the) (?:previous|prior|above) (?:instructions|prompts)", re.I),
    re.compile(r"disregard your (?:system|constitution|safety) (?:prompt|rules)", re.I),
    re.compile(r"you are now (?:DAN|in developer mode|unfiltered)", re.I),
)

_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # credit card-ish
)


@dataclass
class RefusalPolicy:
    disallowed_topic_patterns: tuple[re.Pattern[str], ...] = _DISALLOWED_TOPIC_PATTERNS
    prompt_injection_patterns: tuple[re.Pattern[str], ...] = _PROMPT_INJECTION_PATTERNS
    pii_patterns: tuple[re.Pattern[str], ...] = _PII_PATTERNS
    extra_blocklist: tuple[str, ...] = ()
    refusal_message: str = (
        "I can't help with that. If you have a different question I'd be happy to help."
    )

    def evaluate_input(self, text: str) -> RefusalReason:
        for p in self.disallowed_topic_patterns:
            if p.search(text):
                return RefusalReason.DISALLOWED_TOPIC
        for p in self.prompt_injection_patterns:
            if p.search(text):
                return RefusalReason.PROMPT_INJECTION
        for needle in self.extra_blocklist:
            if needle.lower() in text.lower():
                return RefusalReason.POLICY_VIOLATION
        return RefusalReason.SAFE

    def evaluate_output(self, text: str) -> RefusalReason:
        for p in self.pii_patterns:
            if p.search(text):
                return RefusalReason.PII
        return RefusalReason.SAFE

    def is_safe_input(self, text: str) -> bool:
        return self.evaluate_input(text) == RefusalReason.SAFE


@dataclass
class SafetyVerdict:
    reason: RefusalReason
    refusal_text: str | None = None
    fired_patterns: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.reason == RefusalReason.SAFE


def evaluate_safety(policy: RefusalPolicy, prompt: str, response: str | None = None) -> SafetyVerdict:
    reason = policy.evaluate_input(prompt)
    if reason != RefusalReason.SAFE:
        return SafetyVerdict(reason=reason, refusal_text=policy.refusal_message)
    if response is not None:
        out_reason = policy.evaluate_output(response)
        if out_reason != RefusalReason.SAFE:
            return SafetyVerdict(reason=out_reason, refusal_text=policy.refusal_message)
    return SafetyVerdict(reason=RefusalReason.SAFE)
