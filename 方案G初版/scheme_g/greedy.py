from __future__ import annotations

import random
from bisect import bisect_left
from collections import defaultdict

from .models import CandidateTask, PlannerConfig


def tasks_conflict(
    first: CandidateTask,
    second: CandidateTask,
    config: PlannerConfig,
) -> bool:
    if first.station_id == second.station_id:
        first_before_second = second.start >= first.release_time(config.switch_seconds)
        second_before_first = first.start >= second.release_time(config.switch_seconds)
        if not (first_before_second or second_before_first):
            return True

    if first.target_id == second.target_id:
        first_before_second = second.start >= first.end + config.cooldown
        second_before_first = first.start >= second.end + config.cooldown
        if not (first_before_second or second_before_first):
            return True

    return False


class FeasibleSchedule:
    def __init__(self, config: PlannerConfig):
        self.config = config
        self.tasks: list[CandidateTask] = []
        self.by_station: dict[str, list[CandidateTask]] = defaultdict(list)
        self.by_target: dict[str, list[CandidateTask]] = defaultdict(list)
        self._station_starts: dict[str, list] = defaultdict(list)
        self._target_starts: dict[str, list] = defaultdict(list)
        self.task_ids: set[str] = set()

    def _neighbor_conflict(
        self,
        task: CandidateTask,
        group: list[CandidateTask],
        starts: list,
    ) -> bool:
        """已排任务互不冲突时，只需检查插入位置两侧的任务。"""
        index = bisect_left(starts, task.start)
        if index > 0 and tasks_conflict(task, group[index - 1], self.config):
            return True
        if index < len(group) and tasks_conflict(task, group[index], self.config):
            return True
        return False

    def can_add(self, task: CandidateTask) -> bool:
        if task.task_id in self.task_ids:
            return False
        return not self._neighbor_conflict(
            task,
            self.by_station[task.station_id],
            self._station_starts[task.station_id],
        ) and not self._neighbor_conflict(
            task,
            self.by_target[task.target_id],
            self._target_starts[task.target_id],
        )

    def add(self, task: CandidateTask) -> bool:
        if not self.can_add(task):
            return False
        self.tasks.append(task)
        self.task_ids.add(task.task_id)
        station_starts = self._station_starts[task.station_id]
        station_tasks = self.by_station[task.station_id]
        station_index = bisect_left(station_starts, task.start)
        station_starts.insert(station_index, task.start)
        station_tasks.insert(station_index, task)

        target_starts = self._target_starts[task.target_id]
        target_tasks = self.by_target[task.target_id]
        target_index = bisect_left(target_starts, task.start)
        target_starts.insert(target_index, task.start)
        target_tasks.insert(target_index, task)
        return True


def greedy_schedule(
    tasks: list[CandidateTask],
    config: PlannerConfig,
    mode: str = "ai",
    seed: int | None = None,
) -> list[CandidateTask]:
    candidates = list(tasks)
    if mode == "random":
        random.Random(seed if seed is not None else config.random_seed).shuffle(candidates)
    elif mode == "priority":
        candidates.sort(
            key=lambda item: (
                item.base_score,
                item.task_score / item.occupied_seconds(config.switch_seconds),
                -item.start.timestamp(),
            ),
            reverse=True,
        )
    elif mode == "ai":
        candidates.sort(
            key=lambda item: (
                item.ai_score,
                item.task_score / item.occupied_seconds(config.switch_seconds),
            ),
            reverse=True,
        )
    else:
        raise ValueError(f"未知贪心模式: {mode}")

    schedule = FeasibleSchedule(config)
    covered: set[str] = set()

    # 第一遍优先覆盖尚未出现的目标。
    for task in candidates:
        if task.target_id not in covered and schedule.add(task):
            covered.add(task.target_id)

    # 第二遍允许满足48小时规则的重复计分任务。
    for task in candidates:
        schedule.add(task)

    return sorted(schedule.tasks, key=lambda item: (item.station_id, item.start))
