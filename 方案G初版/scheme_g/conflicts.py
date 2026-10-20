from __future__ import annotations

from .models import CandidateTask, PlannerConfig
from .utils import group_by


Conflict = tuple[str, str]


def build_station_conflicts(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    by_station = group_by(tasks, key=lambda item: item.station_id)

    for group in by_station.values():
        group.sort(key=lambda item: item.start)
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                if second.start >= first.release_time(config.switch_seconds):
                    break
                conflicts.append((first.task_id, second.task_id))
    return conflicts


def build_target_conflicts(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    by_target = group_by(tasks, key=lambda item: item.target_id)

    for group in by_target.values():
        group.sort(key=lambda item: item.start)
        for index, first in enumerate(group):
            for second in group[index + 1 :]:
                if second.start >= first.end + config.cooldown:
                    break
                conflicts.append((first.task_id, second.task_id))
    return conflicts


def build_conflicts(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> tuple[list[Conflict], list[Conflict]]:
    return (
        build_station_conflicts(tasks, config),
        build_target_conflicts(tasks, config),
    )

