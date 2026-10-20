from __future__ import annotations

from scheme_g.evaluator import evaluate_schedule
from scheme_g.features import build_features
from scheme_g.greedy import greedy_schedule
from scheme_g.models import CandidateTask, PlanResult, PlannerConfig
from scheme_g.scorer import LinearTaskScorer
from scheme_g.validator import validate_candidates, validate_schedule


class GreedyPlannerAdapter:
    def __init__(self, config: PlannerConfig, mode: str):
        if mode not in {"random", "priority", "ai"}:
            raise ValueError(mode)
        self.config = config
        self.mode = mode
        self.name = f"{mode}_greedy"

    def plan(self, tasks: list[CandidateTask]) -> PlanResult:
        candidate_validation = validate_candidates(tasks, self.config)
        if not candidate_validation.valid:
            raise ValueError("候选任务不合法:\n" + "\n".join(candidate_validation.errors[:20]))
        build_features(tasks, self.config)
        LinearTaskScorer(self.config.ai_weights).score_tasks(tasks)
        selected = greedy_schedule(
            tasks,
            self.config,
            mode=self.mode,
            seed=self.config.random_seed,
        )
        validation = validate_schedule(selected, self.config)
        return PlanResult(
            algorithm=self.name,
            selected_tasks=selected,
            metrics=evaluate_schedule(selected, self.config),
            solver_status="GREEDY",
            candidate_count=len(tasks),
            filtered_candidate_count=len(tasks),
            validation_errors=validation.errors,
        )

