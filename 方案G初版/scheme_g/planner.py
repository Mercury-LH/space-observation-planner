from __future__ import annotations

from .cpsat_solver import solve_cp_sat
from .evaluator import evaluate_schedule
from .features import build_features
from .filtering import build_reflow_pool, safe_filter_candidates
from .greedy import greedy_schedule
from .local_search import fill_schedule_gaps, improve_single_replacements
from .models import CandidateTask, PlanResult, PlannerConfig
from .scorer import LinearTaskScorer
from .validator import validate_candidates, validate_schedule


class SchemeGPlanner:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def prepare(self, tasks: list[CandidateTask]) -> None:
        validation = validate_candidates(tasks, self.config)
        if not validation.valid:
            preview = "\n".join(validation.errors[:20])
            raise ValueError(f"候选任务数据不合法:\n{preview}")
        build_features(tasks, self.config)
        LinearTaskScorer(self.config.ai_weights).score_tasks(tasks)

    def plan(self, tasks: list[CandidateTask]) -> PlanResult:
        self.prepare(tasks)
        filtered = safe_filter_candidates(tasks, self.config)

        initial = greedy_schedule(filtered, self.config, mode="ai")
        first = solve_cp_sat(
            filtered,
            self.config,
            initial_selected_ids={task.task_id for task in initial},
        )
        if not first.selected_tasks:
            first_selected = initial
        else:
            first_selected = first.selected_tasks

        reflow_pool = build_reflow_pool(
            filtered=filtered,
            all_tasks=tasks,
            selected=first_selected,
            config=self.config,
        )

        final_solver = first
        if len(reflow_pool) > len(filtered):
            reflow_hint = greedy_schedule(reflow_pool, self.config, mode="ai")
            final_solver = solve_cp_sat(
                reflow_pool,
                self.config,
                initial_selected_ids={task.task_id for task in reflow_hint},
            )

        selected = final_solver.selected_tasks or first_selected
        selected = fill_schedule_gaps(selected, tasks, self.config)
        if self.config.local_search_max_removals > 0:
            selected = improve_single_replacements(
                selected,
                reflow_pool,
                self.config,
                max_attempts=self.config.local_search_max_removals,
            )

        validation = validate_schedule(selected, self.config)
        metrics = evaluate_schedule(selected, self.config)
        return PlanResult(
            algorithm="scheme_g_ai_filter_cpsat",
            selected_tasks=selected,
            metrics=metrics,
            solver_status=final_solver.status,
            solver_wall_time=final_solver.wall_time,
            best_bound=final_solver.best_bound,
            candidate_count=len(tasks),
            filtered_candidate_count=len(reflow_pool),
            validation_errors=validation.errors,
        )
