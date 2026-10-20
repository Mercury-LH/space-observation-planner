from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any


PRIORITY_SCORES = {
    "HIGH": 5.0,
    "MEDIUM": 3.0,
    "LOW": 2.0,
}

PRIORITY_ALIASES = {
    "HIGH": "HIGH",
    "H": "HIGH",
    "高": "HIGH",
    "高优": "HIGH",
    "高优先级": "HIGH",
    "MEDIUM": "MEDIUM",
    "M": "MEDIUM",
    "中": "MEDIUM",
    "中优": "MEDIUM",
    "中优先级": "MEDIUM",
    "LOW": "LOW",
    "L": "LOW",
    "低": "LOW",
    "低优": "LOW",
    "低优先级": "LOW",
}


def normalize_priority(value: str) -> str:
    key = value.strip().upper()
    if key not in PRIORITY_ALIASES:
        raise ValueError(f"未知优先级: {value!r}")
    return PRIORITY_ALIASES[key]


def calculate_task_score(base_score: float, duration_seconds: int) -> float:
    if duration_seconds < 120:
        return 0.0
    capped = min(duration_seconds, 240)
    ratio = 0.6 + 0.4 * (capped - 120) / 120
    return base_score * ratio


@dataclass(slots=True)
class CandidateTask:
    task_id: str
    window_id: str
    station_id: str
    target_id: str
    priority: str
    window_start: datetime
    window_end: datetime
    start: datetime
    end: datetime
    base_score: float
    task_score: float
    duration_seconds: int
    min_elevation: float | None = None
    max_elevation: float | None = None
    max_sun_elevation: float | None = None
    ai_score: float = 0.0
    feature_values: dict[str, float] = field(default_factory=dict)

    def release_time(self, switch_seconds: int) -> datetime:
        return self.end + timedelta(seconds=switch_seconds)

    def occupied_seconds(self, switch_seconds: int) -> int:
        return self.duration_seconds + switch_seconds

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ("window_start", "window_end", "start", "end"):
            record[key] = record[key].isoformat().replace("+00:00", "Z")
        record.pop("feature_values", None)
        return record


@dataclass(slots=True)
class PlannerConfig:
    switch_seconds: int = 30
    cooldown_hours: int = 48
    min_observation_seconds: int = 120
    max_observation_seconds: int = 240
    coverage_weight: float = 0.5
    score_weight: float = 0.5
    coverage_normalizer: int = 10000
    score_normalizer: float = 90186.0
    top_k_per_window: int = 4
    high_priority_extra_k: int = 2
    medium_priority_extra_k: int = 1
    reflow_per_target: int = 1
    reflow_per_station_bucket: int = 3
    reflow_bucket_hours: int = 6
    max_reflow_candidates: int = 20000
    solver_time_limit_seconds: float = 60.0
    solver_workers: int = 8
    local_search_max_removals: int = 50
    random_seed: int = 42
    ai_weights: dict[str, float] = field(default_factory=dict)
    ai_weights_file: str | None = None

    @property
    def cooldown(self) -> timedelta:
        return timedelta(hours=self.cooldown_hours)


@dataclass(slots=True)
class PlanMetrics:
    task_count: int
    covered_targets: int
    total_score: float
    normalized_coverage: float
    normalized_score: float
    composite_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanResult:
    algorithm: str
    selected_tasks: list[CandidateTask]
    metrics: PlanMetrics
    solver_status: str = "NOT_RUN"
    solver_wall_time: float = 0.0
    best_bound: float | None = None
    candidate_count: int = 0
    filtered_candidate_count: int = 0
    validation_errors: list[str] = field(default_factory=list)
