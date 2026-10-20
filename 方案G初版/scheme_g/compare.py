from __future__ import annotations

import csv
from pathlib import Path

from .evaluator import evaluate_schedule
from .features import build_features
from .greedy import greedy_schedule
from .models import CandidateTask, PlannerConfig
from .planner import SchemeGPlanner
from .scorer import LinearTaskScorer
from .validator import validate_schedule


def compare_algorithms(
    tasks: list[CandidateTask],
    config: PlannerConfig,
    output_path: str | Path,
) -> list[dict[str, object]]:
    build_features(tasks, config)
    LinearTaskScorer(config.ai_weights).score_tasks(tasks)

    schedules = {
        "random_greedy": greedy_schedule(
            tasks, config, mode="random", seed=config.random_seed
        ),
        "priority_greedy": greedy_schedule(tasks, config, mode="priority"),
        "ai_weighted_greedy": greedy_schedule(tasks, config, mode="ai"),
    }
    scheme_g_result = SchemeGPlanner(config).plan(tasks)
    schedules[scheme_g_result.algorithm] = scheme_g_result.selected_tasks

    rows = []
    for name, schedule in schedules.items():
        metrics = evaluate_schedule(schedule, config)
        validation = validate_schedule(schedule, config)
        rows.append(
            {
                "algorithm": name,
                "task_count": metrics.task_count,
                "covered_targets": metrics.covered_targets,
                "total_score": metrics.total_score,
                "composite_score": metrics.composite_score,
                "valid": validation.valid,
                "error_count": len(validation.errors),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows

