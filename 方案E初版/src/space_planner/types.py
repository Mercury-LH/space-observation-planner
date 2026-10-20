"""Shared records. Times are seconds after the configured UTC start."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math


@dataclass(frozen=True)
class Target:
    target_id: int
    line1: str
    line2: str
    priority: int
    epoch: datetime


@dataclass(frozen=True)
class Station:
    station_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    min_elevation_deg: float = 10.0
    max_elevation_deg: float = 85.0


@dataclass(frozen=True)
class Window:
    target_id: int
    station_id: str
    start: float
    end: float


@dataclass(frozen=True)
class Observation:
    target_id: int
    station_id: str
    start: float
    end: float


@dataclass(frozen=True)
class Rules:
    """Explicit development choices, not an official composite scoring rule.

    Default: one observation per target and lexicographic coverage then score.
    Durations between 120 and 240 receive 60% under conservative 'step';
    'linear' is an alternative assumption. No 48-hour interval is imposed.
    """

    min_duration_s: float = 120.0
    full_duration_s: float = 240.0
    switch_s: float = 30.0
    score_mode: str = "step"
    objective: str = "coverage_then_score"
    repeat_gap_s: float | None = None
    max_observations_per_target: int = 1

    def __post_init__(self):
        for name in ("min_duration_s", "full_duration_s", "switch_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}: {value}")
        if self.min_duration_s <= 0 or self.full_duration_s < self.min_duration_s:
            raise ValueError("Require 0 < minimum duration <= full duration")
        if self.score_mode not in ("step", "linear"):
            raise ValueError("score_mode must be step or linear")
        if self.objective not in ("coverage_then_score", "score_then_coverage"):
            raise ValueError("Unknown development objective")
        if not isinstance(self.max_observations_per_target, int) or self.max_observations_per_target < 1:
            raise ValueError("max_observations_per_target must be a positive integer")
        if self.repeat_gap_s is not None and (
            not math.isfinite(self.repeat_gap_s) or self.repeat_gap_s < 0
        ):
            raise ValueError("repeat_gap_s must be nonnegative or null")
