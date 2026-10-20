from __future__ import annotations

from .evaluator import evaluate_schedule
from .greedy import FeasibleSchedule
from .models import CandidateTask, PlannerConfig


def fill_schedule_gaps(
    selected: list[CandidateTask],
    all_candidates: list[CandidateTask],
    config: PlannerConfig,
    ranked_candidates: list[CandidateTask] | None = None,
) -> list[CandidateTask]:
    """使用完整候选池回填CP-SAT计划空隙，不破坏既有任务。"""
    schedule = FeasibleSchedule(config)
    for task in sorted(selected, key=lambda item: (item.station_id, item.start)):
        if not schedule.add(task):
            raise ValueError(f"CP-SAT结果内部冲突: {task.task_id}")

    covered = {task.target_id for task in schedule.tasks}
    if ranked_candidates is None:
        ranked_candidates = sorted(
            all_candidates,
            key=lambda item: (
                item.ai_score,
                item.task_score / item.occupied_seconds(config.switch_seconds),
            ),
            reverse=True,
        )

    # 先填首次覆盖，再填满足48小时的重复任务。
    for task in ranked_candidates:
        if task.task_id in schedule.task_ids:
            continue
        if task.target_id not in covered and schedule.add(task):
            covered.add(task.target_id)
    for task in ranked_candidates:
        schedule.add(task)

    return sorted(schedule.tasks, key=lambda item: (item.station_id, item.start))


def improve_single_replacements(
    selected: list[CandidateTask],
    all_candidates: list[CandidateTask],
    config: PlannerConfig,
    max_attempts: int = 2000,
) -> list[CandidateTask]:
    """尝试删除一个任务，再用完整候选池贪心回填；只接受综合指标提升。"""
    current = list(selected)
    current_metrics = evaluate_schedule(current, config)
    attempts = 0
    ranked_candidates = sorted(
        all_candidates,
        key=lambda item: (
            item.ai_score,
            item.task_score / item.occupied_seconds(config.switch_seconds),
        ),
        reverse=True,
    )

    # 优先尝试移除AI评分低、占用时间长的任务。
    removal_order = sorted(
        current,
        key=lambda item: (
            item.ai_score,
            -item.occupied_seconds(config.switch_seconds),
        ),
    )

    for removed in removal_order:
        if attempts >= max_attempts:
            break
        attempts += 1
        base = [task for task in current if task.task_id != removed.task_id]
        candidate = fill_schedule_gaps(
            base,
            all_candidates,
            config,
            ranked_candidates=ranked_candidates,
        )
        candidate_metrics = evaluate_schedule(candidate, config)
        if candidate_metrics.composite_score > current_metrics.composite_score + 1e-12:
            current = candidate
            current_metrics = candidate_metrics

    return sorted(current, key=lambda item: (item.station_id, item.start))
