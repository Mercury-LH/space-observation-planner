from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .models import (
    PRIORITY_SCORES,
    CandidateTask,
    PlanMetrics,
    PlannerConfig,
    calculate_task_score,
    normalize_priority,
)
from .utils import parse_datetime


def load_config(path: str | Path) -> PlannerConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(PlannerConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"配置包含未知字段: {sorted(unknown)}")

    config = PlannerConfig(**raw)
    if config.ai_weights_file:
        weights_path = Path(config.ai_weights_file)
        if not weights_path.is_absolute():
            weights_path = config_path.parent / weights_path
        config.ai_weights = json.loads(weights_path.read_text(encoding="utf-8"))
    return config


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return float(value) if value else None


def load_candidates(path: str | Path) -> list[CandidateTask]:
    candidate_path = Path(path)
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "task_id",
            "window_id",
            "station_id",
            "target_id",
            "priority",
            "window_start",
            "window_end",
            "start",
            "end",
            "base_score",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"候选任务文件缺少字段: {sorted(missing)}")

        result: list[CandidateTask] = []
        seen_ids: set[str] = set()

        for line_number, row in enumerate(reader, start=2):
            task_id = row["task_id"].strip()
            if task_id in seen_ids:
                raise ValueError(f"第{line_number}行任务编号重复: {task_id}")
            seen_ids.add(task_id)

            priority = normalize_priority(row["priority"])
            start = parse_datetime(row["start"])
            end = parse_datetime(row["end"])
            duration = int(row.get("duration_seconds") or (end - start).total_seconds())
            base_score = float(row.get("base_score") or PRIORITY_SCORES[priority])
            score_text = row.get("task_score", "").strip()
            task_score = (
                float(score_text)
                if score_text
                else calculate_task_score(base_score, duration)
            )

            result.append(
                CandidateTask(
                    task_id=task_id,
                    window_id=row["window_id"].strip(),
                    station_id=row["station_id"].strip(),
                    target_id=row["target_id"].strip(),
                    priority=priority,
                    window_start=parse_datetime(row["window_start"]),
                    window_end=parse_datetime(row["window_end"]),
                    start=start,
                    end=end,
                    base_score=base_score,
                    task_score=task_score,
                    duration_seconds=duration,
                    min_elevation=_optional_float(row, "min_elevation"),
                    max_elevation=_optional_float(row, "max_elevation"),
                    max_sun_elevation=_optional_float(row, "max_sun_elevation"),
                )
            )
    return result


def write_schedule(path: str | Path, tasks: list[CandidateTask]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [task.to_record() for task in sorted(tasks, key=lambda x: (x.station_id, x.start))]
    fieldnames = [
        "task_id",
        "window_id",
        "station_id",
        "target_id",
        "priority",
        "window_start",
        "window_end",
        "start",
        "end",
        "base_score",
        "task_score",
        "duration_seconds",
        "min_elevation",
        "max_elevation",
        "max_sun_elevation",
        "ai_score",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_metrics(path: str | Path, metrics: PlanMetrics, extra: dict[str, Any]) -> None:
    payload = metrics.to_dict()
    payload.update(extra)
    write_json(path, payload)

