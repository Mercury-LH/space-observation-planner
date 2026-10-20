from __future__ import annotations

from collections import defaultdict
from .models import CandidateTask, PlannerConfig
from .utils import group_by


def safe_filter_candidates(
    tasks: list[CandidateTask],
    config: PlannerConfig,
    preserve_ids: set[str] | None = None,
) -> list[CandidateTask]:
    """AI Top-K加窗口、时长和目标保底，不做全局粗暴截断。"""
    if not tasks:
        return []

    preserve_ids = set(preserve_ids or ())
    retained = set(preserve_ids)

    by_window = group_by(tasks, key=lambda item: item.window_id)
    for group in by_window.values():
        priority = group[0].priority
        extra = 0
        if priority == "HIGH":
            extra = config.high_priority_extra_k
        elif priority == "MEDIUM":
            extra = config.medium_priority_extra_k
        k = config.top_k_per_window + extra

        ranked = sorted(group, key=lambda item: item.ai_score, reverse=True)
        retained.update(task.task_id for task in ranked[:k])

        retained.add(min(group, key=lambda item: item.start).task_id)
        retained.add(max(group, key=lambda item: item.start).task_id)

        by_duration = group_by(group, key=lambda item: item.duration_seconds)
        for duration_group in by_duration.values():
            retained.add(max(duration_group, key=lambda item: item.ai_score).task_id)

    by_target = group_by(tasks, key=lambda item: item.target_id)
    for group in by_target.values():
        retained.add(max(group, key=lambda item: item.ai_score).task_id)
        retained.add(min(group, key=lambda item: item.start).task_id)
        retained.add(max(group, key=lambda item: item.start).task_id)

    return [task for task in tasks if task.task_id in retained]


def build_reflow_pool(
    filtered: list[CandidateTask],
    all_tasks: list[CandidateTask],
    selected: list[CandidateTask],
    config: PlannerConfig,
) -> list[CandidateTask]:
    """把AI筛掉但可能有局部价值的任务重新送入第二轮CP-SAT。"""
    retained = {task.task_id for task in filtered}
    retained.update(task.task_id for task in selected)
    omitted = [task for task in all_tasks if task.task_id not in retained]
    omitted.sort(key=lambda item: item.ai_score, reverse=True)

    by_target = group_by(omitted, key=lambda item: item.target_id)
    for group in by_target.values():
        for task in group[: config.reflow_per_target]:
            retained.add(task.task_id)

    horizon_start = min(task.start for task in all_tasks)
    bucket_seconds = config.reflow_bucket_hours * 3600
    by_bucket: dict[tuple[str, int], list[CandidateTask]] = defaultdict(list)
    for task in omitted:
        bucket = int((task.start - horizon_start).total_seconds() // bucket_seconds)
        by_bucket[(task.station_id, bucket)].append(task)

    added = 0
    for group in by_bucket.values():
        group.sort(key=lambda item: item.ai_score, reverse=True)
        for task in group[: config.reflow_per_station_bucket]:
            if added >= config.max_reflow_candidates:
                break
            if task.task_id not in retained:
                retained.add(task.task_id)
                added += 1
        if added >= config.max_reflow_candidates:
            break

    return [task for task in all_tasks if task.task_id in retained]
