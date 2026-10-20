from __future__ import annotations

from dataclasses import dataclass

from .models import CandidateTask, PlannerConfig
from .utils import group_by


@dataclass(slots=True)
class SolverOutput:
    selected_tasks: list[CandidateTask]
    status: str
    wall_time: float
    best_bound: float | None
    objective_value: float | None


def _objective_coefficients(config: PlannerConfig) -> tuple[int, int, int]:
    objective_scale = 1_000_000_000
    task_score_scale = 1000
    coverage_coefficient = max(
        1,
        round(
            objective_scale
            * config.coverage_weight
            / max(config.coverage_normalizer, 1)
        ),
    )
    score_coefficient = max(
        1,
        round(
            objective_scale
            * config.score_weight
            / max(config.score_normalizer * task_score_scale, 1.0)
        ),
    )
    return coverage_coefficient, score_coefficient, task_score_scale


def _add_no_overlap_constraints(model, tasks, variables, config) -> None:
    """用全局NoOverlap约束替代海量两两冲突约束。

    每个候选任务的开始时刻固定，presence由对应布尔变量控制。站点资源的
    区间延长30秒表示转换时间；目标资源的区间延长48小时表示再次观测
    冷却期。该表示与原来的两两冲突判断等价，但模型规模从冲突对数量降为
    候选任务数量，适合大规模候选集。
    """
    origin = min(task.start for task in tasks)
    station_intervals = {}
    target_intervals = {}

    for index, task in enumerate(tasks):
        start = round((task.start - origin).total_seconds())
        station_size = task.duration_seconds + config.switch_seconds
        target_size = task.duration_seconds + round(config.cooldown.total_seconds())
        presence = variables[task.task_id]

        station_interval = model.NewOptionalIntervalVar(
            start,
            station_size,
            start + station_size,
            presence,
            f"station_interval_{index}",
        )
        target_interval = model.NewOptionalIntervalVar(
            start,
            target_size,
            start + target_size,
            presence,
            f"target_interval_{index}",
        )
        station_intervals.setdefault(task.station_id, []).append(station_interval)
        target_intervals.setdefault(task.target_id, []).append(target_interval)

    for intervals in station_intervals.values():
        model.AddNoOverlap(intervals)
    for intervals in target_intervals.values():
        model.AddNoOverlap(intervals)


def solve_cp_sat(
    tasks: list[CandidateTask],
    config: PlannerConfig,
    initial_selected_ids: set[str] | None = None,
    time_limit_seconds: float | None = None,
) -> SolverOutput:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError(
            "缺少OR-Tools。请执行: python -m pip install -r scheme_g/requirements.txt"
        ) from exc

    if not tasks:
        return SolverOutput([], "EMPTY", 0.0, 0.0, 0.0)

    model = cp_model.CpModel()
    variables = {
        task.task_id: model.NewBoolVar(f"task_{index}")
        for index, task in enumerate(tasks)
    }

    _add_no_overlap_constraints(model, tasks, variables, config)

    tasks_by_target = group_by(tasks, key=lambda item: item.target_id)
    coverage_variables = {}
    for index, (target_id, target_tasks) in enumerate(tasks_by_target.items()):
        covered = model.NewBoolVar(f"covered_{index}")
        coverage_variables[target_id] = covered
        selected_sum = sum(variables[task.task_id] for task in target_tasks)
        model.Add(covered <= selected_sum)
        for task in target_tasks:
            model.Add(covered >= variables[task.task_id])

    coverage_coefficient, score_coefficient, task_score_scale = _objective_coefficients(config)
    coverage_term = coverage_coefficient * sum(coverage_variables.values())
    score_term = score_coefficient * sum(
        round(task.task_score * task_score_scale) * variables[task.task_id]
        for task in tasks
    )
    model.Maximize(coverage_term + score_term)

    initial_selected_ids = set(initial_selected_ids or ())
    for task in tasks:
        model.AddHint(
            variables[task.task_id],
            1 if task.task_id in initial_selected_ids else 0,
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = (
        time_limit_seconds
        if time_limit_seconds is not None
        else config.solver_time_limit_seconds
    )
    solver.parameters.num_search_workers = config.solver_workers
    solver.parameters.random_seed = config.random_seed

    status = solver.Solve(model)
    feasible_statuses = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    selected = []
    if status in feasible_statuses:
        selected = [
            task
            for task in tasks
            if solver.Value(variables[task.task_id]) == 1
        ]

    return SolverOutput(
        selected_tasks=sorted(selected, key=lambda item: (item.station_id, item.start)),
        status=solver.StatusName(status),
        wall_time=solver.WallTime(),
        best_bound=(solver.BestObjectiveBound() if status in feasible_statuses else None),
        objective_value=(solver.ObjectiveValue() if status in feasible_statuses else None),
    )
