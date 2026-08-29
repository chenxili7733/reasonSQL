"""Distillation-coefficient schedules for the optional veRL v0.8.0 patch."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OPDSchedule:
    start: float = 1.0
    end: float = 0.2
    kind: str = "cosine"
    warmup_steps: int = 0

    def validate(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("OPD coefficients must be non-negative")
        if self.kind not in {"constant", "linear", "cosine"}:
            raise ValueError(f"unsupported OPD schedule: {self.kind}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

    def coefficient(self, step: int, total_steps: int) -> float:
        self.validate()
        if self.kind == "constant" or total_steps <= 1:
            return self.start
        effective_step = max(0, int(step) - self.warmup_steps)
        effective_total = max(1, int(total_steps) - self.warmup_steps)
        progress = min(1.0, effective_step / effective_total)
        if self.kind == "linear":
            mix = progress
        else:
            mix = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.start + (self.end - self.start) * mix


def schedule_from_env() -> OPDSchedule:
    return OPDSchedule(
        start=float(os.environ.get("REASONSQL_OPD_LAMBDA_START", "1.0")),
        end=float(os.environ.get("REASONSQL_OPD_LAMBDA_END", "0.2")),
        kind=os.environ.get("REASONSQL_OPD_SCHEDULE", "cosine").strip().casefold(),
        warmup_steps=int(os.environ.get("REASONSQL_OPD_WARMUP_STEPS", "0")),
    )


def coefficient_from_env(step: int, total_steps: int) -> float:
    return schedule_from_env().coefficient(step, total_steps)

