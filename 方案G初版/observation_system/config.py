from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scheme_g.models import PlannerConfig
from scheme_g.utils import parse_datetime


@dataclass(slots=True)
class DataConfig:
    tle_path: str
    station_path: str
    priority_dir: str


@dataclass(slots=True)
class ScenarioConfig:
    start_time: datetime
    end_time: datetime
    minimum_elevation_deg: float = 10.0
    maximum_elevation_deg: float = 85.0
    maximum_sun_elevation_deg: float = -6.0
    minimum_window_seconds: int = 120


@dataclass(slots=True)
class VisibilityConfig:
    sample_step_seconds: int = 20
    boundary_tolerance_seconds: float = 0.25
    workers: int = 1
    target_limit: int | None = None
    station_limit: int | None = None


@dataclass(slots=True)
class CandidateConfig:
    durations_seconds: tuple[int, ...] = (120, 240)
    grid_seconds: int = 30
    max_starts_per_window_duration: int = 20


@dataclass(slots=True)
class OutputConfig:
    directory: str = "outputs/full_pipeline"
    cache_windows: bool = True


@dataclass(slots=True)
class SystemConfig:
    data: DataConfig
    scenario: ScenarioConfig
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    planner_name: str = "scheme_g"
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_system_config(path: str | Path) -> SystemConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent.parent

    data_raw = raw["data"]
    data = DataConfig(
        tle_path=_resolve_path(base, data_raw["tle_path"]),
        station_path=_resolve_path(base, data_raw["station_path"]),
        priority_dir=_resolve_path(base, data_raw["priority_dir"]),
    )

    scenario_raw = raw["scenario"]
    scenario = ScenarioConfig(
        start_time=parse_datetime(scenario_raw["start_time"]),
        end_time=parse_datetime(scenario_raw["end_time"]),
        minimum_elevation_deg=float(scenario_raw.get("minimum_elevation_deg", 10.0)),
        maximum_elevation_deg=float(scenario_raw.get("maximum_elevation_deg", 85.0)),
        maximum_sun_elevation_deg=float(
            scenario_raw.get("maximum_sun_elevation_deg", -6.0)
        ),
        minimum_window_seconds=int(scenario_raw.get("minimum_window_seconds", 120)),
    )

    visibility_raw = raw.get("visibility", {})
    visibility = VisibilityConfig(**visibility_raw)

    candidate_raw = dict(raw.get("candidates", {}))
    if "durations_seconds" in candidate_raw:
        candidate_raw["durations_seconds"] = tuple(candidate_raw["durations_seconds"])
    candidates = CandidateConfig(**candidate_raw)

    planner_raw = raw.get("planner", {})
    planner = PlannerConfig(**planner_raw)

    output_raw = dict(raw.get("output", {}))
    if "directory" in output_raw:
        output_raw["directory"] = _resolve_path(base, output_raw["directory"])
    output = OutputConfig(**output_raw)

    if scenario.end_time <= scenario.start_time:
        raise ValueError("规划结束时间必须晚于开始时间")
    return SystemConfig(
        data=data,
        scenario=scenario,
        visibility=visibility,
        candidates=candidates,
        planner_name=raw.get("planner_name", "scheme_g"),
        planner=planner,
        output=output,
    )
