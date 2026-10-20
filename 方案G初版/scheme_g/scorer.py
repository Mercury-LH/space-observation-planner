from __future__ import annotations

from .models import CandidateTask


DEFAULT_WEIGHTS = {
    "priority": 1.2,
    "task_score": 1.0,
    "rarity": 2.4,
    "urgency": 1.8,
    "efficiency": 1.2,
    "station_flexibility": -0.4,
    "conflict_pressure": -0.8,
    "occupation": -0.3,
}


class LinearTaskScorer:
    """可由遗传搜索训练权重的可解释AI价值模型。"""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def predict(self, feature_values: dict[str, float]) -> float:
        return sum(
            self.weights.get(name, 0.0) * value
            for name, value in feature_values.items()
        )

    def score_tasks(self, tasks: list[CandidateTask]) -> None:
        for task in tasks:
            task.ai_score = self.predict(task.feature_values)

