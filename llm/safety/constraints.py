"""Constitutional constraints.

A *constraint* is a structured, machine-checkable predicate that runs at
inference time. Examples:

* ``length_within(min, max)`` — output length budget.
* ``no_url(domain_blocklist)`` — strip / refuse responses citing blocklisted URLs.
* ``json_schema(schema)`` — output must validate against a JSON schema.
* ``regex_must_not_match(pattern)`` — generic regex blacklist.

The :class:`ConstraintEvaluator` runs all attached constraints and returns the
aggregated verdict, optionally short-circuiting on the first failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


@dataclass
class ConstraintResult:
    passed: bool
    reason: str = ""


@dataclass
class ConstitutionalConstraint:
    name: str
    check: Callable[[str], ConstraintResult]
    severity: str = "block"  # "block" | "warn"


@dataclass
class ConstraintEvaluator:
    constraints: list[ConstitutionalConstraint] = field(default_factory=list)
    short_circuit: bool = True

    def add(self, constraint: ConstitutionalConstraint) -> None:
        self.constraints.append(constraint)

    def evaluate(self, text: str) -> list[tuple[ConstitutionalConstraint, ConstraintResult]]:
        results: list[tuple[ConstitutionalConstraint, ConstraintResult]] = []
        for c in self.constraints:
            result = c.check(text)
            results.append((c, result))
            if not result.passed and self.short_circuit and c.severity == "block":
                break
        return results

    def is_acceptable(self, text: str) -> bool:
        for c, r in self.evaluate(text):
            if not r.passed and c.severity == "block":
                return False
        return True


# ---------------------------------------------------------------------- #
# Built-in constraints                                                    #
# ---------------------------------------------------------------------- #
def length_within(min_chars: int, max_chars: int) -> ConstitutionalConstraint:
    def check(text: str) -> ConstraintResult:
        n = len(text)
        if n < min_chars:
            return ConstraintResult(False, f"too short ({n} < {min_chars})")
        if n > max_chars:
            return ConstraintResult(False, f"too long ({n} > {max_chars})")
        return ConstraintResult(True)
    return ConstitutionalConstraint(name="length_within", check=check)


def regex_must_not_match(pattern: str | re.Pattern[str], label: str = "regex") -> ConstitutionalConstraint:
    p = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, re.IGNORECASE)

    def check(text: str) -> ConstraintResult:
        if p.search(text):
            return ConstraintResult(False, f"matched {p.pattern!r}")
        return ConstraintResult(True)

    return ConstitutionalConstraint(name=f"no_{label}", check=check)


def no_urls(blocked_domains: Iterable[str]) -> ConstitutionalConstraint:
    blocked = tuple(d.lower() for d in blocked_domains)

    def check(text: str) -> ConstraintResult:
        for d in blocked:
            if d in text.lower():
                return ConstraintResult(False, f"blocked domain {d}")
        return ConstraintResult(True)

    return ConstitutionalConstraint(name="no_blocked_urls", check=check)


def json_schema_validator(schema: dict) -> ConstitutionalConstraint:
    def check(text: str) -> ConstraintResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return ConstraintResult(False, f"invalid JSON: {exc}")
        # Minimal type check; for full validation use jsonschema in caller env.
        if "type" in schema and not _matches_type(data, schema["type"]):
            return ConstraintResult(False, f"expected {schema['type']}")
        return ConstraintResult(True)

    return ConstitutionalConstraint(name="json_schema", check=check)


def _matches_type(value, type_name: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)),
        "integer": isinstance(value, int),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(type_name, True)
