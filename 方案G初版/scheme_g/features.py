from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from .models import CandidateTask, PlannerConfig
from .utils import group_by


def build_features(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> dict[str, dict[str, float]]:
    if not tasks:
        return {}

    horizon_start = min(task.window_start for task in tasks)
    horizon_end = max(task.window_end for task in tasks)
    horizon_seconds = max((horizon_end - horizon_start).total_seconds(), 1.0)

    windows_by_target: dict[str, set[str]] = defaultdict(set)
    stations_by_target: dict[str, set[str]] = defaultdict(set)
    target_last_window: dict[str, datetime] = {}
    candidates_per_target = Counter(task.target_id for task in tasks)

    bucket_seconds = 600
    station_bucket_load: Counter[tuple[str, int]] = Counter()

    for task in tasks:
        windows_by_target[task.target_id].add(task.window_id)
        stations_by_target[task.target_id].add(task.station_id)
        target_last_window[task.target_id] = max(
            task.window_end,
            target_last_window.get(task.target_id, task.window_end),
        )
        bucket = int((task.start - horizon_start).total_seconds() // bucket_seconds)
        station_bucket_load[(task.station_id, bucket)] += 1

    tasks_by_target = group_by(tasks, key=lambda item: item.target_id)
    future_window_counts: dict[str, dict[str, int]] = {}
    for target_id, group in tasks_by_target.items():
        ordered_windows = sorted(
            {(task.window_start, task.window_id) for task in group},
            key=lambda item: item[0],
        )
        future_window_counts[target_id] = {
            window_id: len(ordered_windows) - index - 1
            for index, (_, window_id) in enumerate(ordered_windows)
        }

    features: dict[str, dict[str, float]] = {}
    max_station_flexibility = max(len(value) for value in stations_by_target.values())

    for task in tasks:
        bucket = int((task.start - horizon_start).total_seconds() // bucket_seconds)
        future_count = future_window_counts[task.target_id].get(task.window_id, 0)
        rarity = 1.0 / (1.0 + len(windows_by_target[task.target_id]))
        urgency = 1.0 / (1.0 + future_count)
        station_flexibility = (
            len(stations_by_target[task.target_id]) / max_station_flexibility
            if max_station_flexibility
            else 0.0
        )
        conflict_pressure = station_bucket_load[(task.station_id, bucket)]
        efficiency = task.task_score / max(task.occupied_seconds(config.switch_seconds), 1)
        occupation = task.occupied_seconds(config.switch_seconds) / (
            config.max_observation_seconds + config.switch_seconds
        )
        planning_progress = (task.start - horizon_start).total_seconds() / horizon_seconds
        last_opportunity_seconds = max(
            (target_last_window[task.target_id] - task.start).total_seconds(),
            0.0,
        )

        values = {
            "priority": task.base_score / 5.0,
            "task_score": task.task_score / 5.0,
            "rarity": rarity,
            "urgency": urgency,
            "efficiency": efficiency * 100.0,
            "station_flexibility": station_flexibility,
            "conflict_pressure": conflict_pressure / 100.0,
            "occupation": occupation,
            "planning_progress": planning_progress,
            "target_candidate_rarity": 1.0 / (1.0 + candidates_per_target[task.target_id]),
            "last_opportunity_hours": last_opportunity_seconds / 3600.0,
        }
        task.feature_values = values
        features[task.task_id] = values

    return features

