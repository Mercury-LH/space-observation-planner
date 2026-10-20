from __future__ import annotations

from .models import CandidateTask, PlanMetrics, PlannerConfig


def evaluate_schedule(
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> PlanMetrics:
    covered_targets = len({task.target_id for task in tasks})
    total_score = sum(task.task_score for task in tasks)
    normalized_coverage = covered_targets / max(config.coverage_normalizer, 1)
    normalized_score = total_score / max(config.score_normalizer, 1e-9)
    composite = (
        config.coverage_weight * normalized_coverage
        + config.score_weight * normalized_score
    )
    return PlanMetrics(
        task_count=len(tasks),
        covered_targets=covered_targets,
        total_score=round(total_score, 6),
        normalized_coverage=normalized_coverage,
        normalized_score=normalized_score,
        composite_score=composite,
    )

