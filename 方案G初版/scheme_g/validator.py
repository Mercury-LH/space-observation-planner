from __future__ import annotations

from dataclasses import dataclass

from .models import CandidateTask, PlannerConfig, calculate_task_score
from .utils import group_by


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def validate_candidates(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> ValidationResult:
    errors: list[str] = []

    for task in tasks:
        actual_duration = round((task.end - task.start).total_seconds())
        if actual_duration != task.duration_seconds:
            errors.append(
                f"任务{task.task_id}: duration_seconds={task.duration_seconds}，"
                f"但开始结束时间计算为{actual_duration}"
            )
        if not (
            config.min_observation_seconds
            <= task.duration_seconds
            <= config.max_observation_seconds
        ):
            errors.append(f"任务{task.task_id}: 观测时长不合法")
        if task.start < task.window_start or task.end > task.window_end:
            errors.append(f"任务{task.task_id}: 未完全落在合法可见窗口内")

        expected_score = calculate_task_score(task.base_score, task.duration_seconds)
        if abs(expected_score - task.task_score) > 1e-6:
            errors.append(
                f"任务{task.task_id}: 得分{task.task_score}与规则复算"
                f"{expected_score}不一致"
            )

        if task.min_elevation is not None and task.min_elevation < 10.0 - 1e-9:
            errors.append(f"任务{task.task_id}: 最低仰角低于10度")
        if task.max_elevation is not None and task.max_elevation > 85.0 + 1e-9:
            errors.append(f"任务{task.task_id}: 最高仰角超过85度")
        if (
            task.max_sun_elevation is not None
            and task.max_sun_elevation >= -6.0
        ):
            errors.append(f"任务{task.task_id}: 太阳高度角不低于-6度")

    return ValidationResult(valid=not errors, errors=errors)


def validate_schedule(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> ValidationResult:
    errors = list(validate_candidates(tasks, config).errors)

    by_station = group_by(tasks, key=lambda item: item.station_id)
    for station_id, group in by_station.items():
        group.sort(key=lambda item: item.start)
        for first, second in zip(group, group[1:]):
            required_start = first.release_time(config.switch_seconds)
            if second.start < required_start:
                errors.append(
                    f"站{station_id}: 任务{first.task_id}与{second.task_id}冲突，"
                    f"未满足{config.switch_seconds}秒切换"
                )

    by_target = group_by(tasks, key=lambda item: item.target_id)
    for target_id, group in by_target.items():
        group.sort(key=lambda item: item.start)
        for first, second in zip(group, group[1:]):
            required_start = first.end + config.cooldown
            if second.start < required_start:
                errors.append(
                    f"目标{target_id}: 任务{first.task_id}与{second.task_id}"
                    f"未满足{config.cooldown_hours}小时重复观测间隔"
                )

    return ValidationResult(valid=not errors, errors=errors)
