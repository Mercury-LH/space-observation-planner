from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scheme_g.io import write_metrics, write_schedule
from scheme_g.models import CandidateTask, PlanResult

from .data_loader import DataAudit
from .domain import ObservationWindow


def build_detailed_statistics(tasks: list[CandidateTask]) -> dict[str, Any]:
    priority_task_counts = Counter(task.priority for task in tasks)
    priority_targets: dict[str, set[str]] = defaultdict(set)
    station_task_counts = Counter(task.station_id for task in tasks)
    station_observation_seconds = Counter()
    for task in tasks:
        priority_targets[task.priority].add(task.target_id)
        station_observation_seconds[task.station_id] += task.duration_seconds

    return {
        "priority_task_counts": dict(priority_task_counts),
        "priority_covered_targets": {
            priority: len(targets)
            for priority, targets in priority_targets.items()
        },
        "station_task_counts": dict(station_task_counts),
        "station_observation_seconds": dict(station_observation_seconds),
    }


def export_pipeline_result(
    output_dir: str | Path,
    result: PlanResult,
    audit: DataAudit,
    windows: list[ObservationWindow],
    candidate_count: int,
    visibility_failures: list[str],
    elapsed_seconds: dict[str, float],
    active_target_count: int | None = None,
    active_station_count: int | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_schedule(output / "schedule.csv", result.selected_tasks)
    write_metrics(
        output / "metrics.json",
        result.metrics,
        {
            "algorithm": result.algorithm,
            "solver_status": result.solver_status,
            "solver_wall_time": result.solver_wall_time,
            "best_bound": result.best_bound,
            "input_target_count": audit.target_count,
            "input_station_count": audit.station_count,
            "active_target_count": active_target_count or audit.target_count,
            "active_station_count": active_station_count or audit.station_count,
            "window_count": len(windows),
            "candidate_count": candidate_count,
            "planner_candidate_count": result.filtered_candidate_count,
            "data_audit_valid": audit.valid,
            "visibility_failure_count": len(visibility_failures),
            "validation_errors": result.validation_errors,
            "timing_seconds": elapsed_seconds,
            "details": build_detailed_statistics(result.selected_tasks),
        },
    )
    (output / "visibility_failures.json").write_text(
        json.dumps(visibility_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
