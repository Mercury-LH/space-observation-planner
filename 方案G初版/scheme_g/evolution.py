from __future__ import annotations

import random
from dataclasses import dataclass

from .evaluator import evaluate_schedule
from .features import build_features
from .greedy import greedy_schedule
from .models import CandidateTask, PlannerConfig
from .scorer import DEFAULT_WEIGHTS, LinearTaskScorer


WEIGHT_RANGES = {
    "priority": (-1.0, 5.0),
    "task_score": (-1.0, 5.0),
    "rarity": (-1.0, 8.0),
    "urgency": (-1.0, 8.0),
    "efficiency": (-2.0, 5.0),
    "station_flexibility": (-5.0, 3.0),
    "conflict_pressure": (-5.0, 2.0),
    "occupation": (-5.0, 2.0),
}


@dataclass(slots=True)
class Individual:
    weights: dict[str, float]
    fitness: float = float("-inf")


def _random_weights(rng: random.Random) -> dict[str, float]:
    return {
        name: rng.uniform(low, high)
        for name, (low, high) in WEIGHT_RANGES.items()
    }


def _evaluate(
    individual: Individual,
    tasks: list[CandidateTask],
    config: PlannerConfig,
) -> float:
    LinearTaskScorer(individual.weights).score_tasks(tasks)
    schedule = greedy_schedule(tasks, config, mode="ai")
    return evaluate_schedule(schedule, config).composite_score


def evolve_weights(
    tasks: list[CandidateTask],
    config: PlannerConfig,
    population_size: int = 24,
    generations: int = 20,
) -> tuple[dict[str, float], list[float]]:
    if population_size < 4:
        raise ValueError("population_size至少为4")
    rng = random.Random(config.random_seed)
    build_features(tasks, config)

    population = [Individual(_random_weights(rng)) for _ in range(population_size - 1)]
    population.append(Individual(dict(DEFAULT_WEIGHTS)))
    history: list[float] = []

    for _ in range(generations):
        for individual in population:
            individual.fitness = _evaluate(individual, tasks, config)
        population.sort(key=lambda item: item.fitness, reverse=True)
        history.append(population[0].fitness)

        elite_count = max(2, population_size // 4)
        elites = population[:elite_count]
        next_population = [Individual(dict(item.weights), item.fitness) for item in elites]

        while len(next_population) < population_size:
            first, second = rng.sample(elites, 2)
            child = {}
            for name, (low, high) in WEIGHT_RANGES.items():
                value = first.weights[name] if rng.random() < 0.5 else second.weights[name]
                if rng.random() < 0.25:
                    value += rng.gauss(0.0, (high - low) * 0.08)
                child[name] = min(high, max(low, value))
            next_population.append(Individual(child))
        population = next_population

    for individual in population:
        individual.fitness = _evaluate(individual, tasks, config)
    best = max(population, key=lambda item: item.fitness)
    return best.weights, history

