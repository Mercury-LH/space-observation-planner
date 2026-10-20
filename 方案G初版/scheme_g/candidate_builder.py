from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .models import (
    PRIORITY_SCORES,
    CandidateTask,
    calculate_task_score,
    normalize_priority,
)


@dataclass(slots=True)
class VisibilityWindow:
    window_id: str
    station_id: str
    target_id: str
    priority: str
    start: datetime
    end: datetime
    base_score: float | None = None
    min_elevation: float | None = None
    max_elevation: float | None = None
    max_sun_elevation: float | None = None


def _ceil_to_grid(value: datetime, step_seconds: int) -> datetime:
    timestamp = value.astimezone(timezone.utc).timestamp()
    aligned = math.ceil(timestamp / step_seconds) * step_seconds
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit <= 1:
        return [0]
    return sorted(
        {
            round(index * (count - 1) / (limit - 1))
            for index in range(limit)
        }
    )


def build_candidates_from_windows(
    windows: list[VisibilityWindow],
    durations_seconds: tuple[int, ...] = (120, 240),
    grid_seconds: int = 30,
    max_starts_per_window_duration: int = 20,
) -> list[CandidateTask]:
    """将前序轨道模块给出的合法窗口转为方案G候选任务。"""
    if grid_seconds <= 0:
        raise ValueError("grid_seconds必须大于0")
    if max_starts_per_window_duration <= 0:
        raise ValueError("max_starts_per_window_duration必须大于0")

    result: list[CandidateTask] = []
    for window in windows:
        priority = normalize_priority(window.priority)
        base_score = (
            window.base_score
            if window.base_score is not None
            else PRIORITY_SCORES[priority]
        )
        first_grid_start = _ceil_to_grid(window.start, grid_seconds)

        for duration in durations_seconds:
            if duration < 120 or duration > 240:
                raise ValueError(f"候选观测时长必须在120到240秒内: {duration}")
            latest_start = window.end - timedelta(seconds=duration)
            if first_grid_start > latest_start:
                continue

            count = int((latest_start - first_grid_start).total_seconds() // grid_seconds) + 1
            for position in _sample_indices(count, max_starts_per_window_duration):
                start = first_grid_start + timedelta(seconds=position * grid_seconds)
                end = start + timedelta(seconds=duration)
                task_id = f"{window.window_id}:{duration}:{position}"
                result.append(
                    CandidateTask(
                        task_id=task_id,
                        window_id=window.window_id,
                        station_id=window.station_id,
                        target_id=window.target_id,
                        priority=priority,
                        window_start=window.start,
                        window_end=window.end,
                        start=start,
                        end=end,
                        base_score=base_score,
                        task_score=calculate_task_score(base_score, duration),
                        duration_seconds=duration,
                        min_elevation=window.min_elevation,
                        max_elevation=window.max_elevation,
                        max_sun_elevation=window.max_sun_elevation,
                    )
                )
    return result

