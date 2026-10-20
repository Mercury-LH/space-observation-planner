"""Continuous-window greedy and adaptive large neighbourhood search baselines.

Times are seconds from the configured UTC epoch. No start-time grid or window
pruning is used: insertion examines every physical window and every intersecting
gap in the current station schedule. Gap endpoints suffice to detect an
insertion with the other observations fixed; they do NOT enumerate all possible
joint rearrangements or certify a globally optimal schedule.

The default one-observation limit and lexicographic objective are explicit
development choices in ``Rules``, not confirmed competition rules. When repeats
are enabled, their scores add, coverage remains distinct-target coverage, and
the optional repeat interval is measured start to start, across stations.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math
import random
import time
from typing import Iterable

from .types import Observation, Rules, Window


_EPS = 1e-7


def observation_score(observation: Observation, priority: int, rules: Rules) -> float:
    """Development scoring: confirmed endpoints, configurable interior rule."""
    duration = observation.end - observation.start
    if duration + _EPS < rules.min_duration_s:
        return 0.0
    if duration + _EPS >= rules.full_duration_s:
        return float(priority)
    if rules.score_mode == "linear":
        fraction = (duration - rules.min_duration_s) / (
            rules.full_duration_s - rules.min_duration_s
        )
        return float(priority) * (0.6 + 0.4 * fraction)
    return float(priority) * 0.6


def objective_metrics(
    plan: Iterable[Observation], priorities: dict[int, int], rules: Rules
) -> dict:
    """Report coverage and additive observation score separately."""
    plan = list(plan)
    covered = {observation.target_id for observation in plan}
    return {
        "coverage_count": len(covered),
        "observation_count": len(plan),
        "observation_score": sum(
            observation_score(observation, priorities[observation.target_id], rules)
            for observation in plan
        ),
        "coverage_by_priority": {
            str(priority): sum(priorities[target] == priority for target in covered)
            for priority in sorted(set(priorities.values()), reverse=True)
        },
        "observing_seconds": sum(item.end - item.start for item in plan),
    }


def _objective(plan: Iterable[Observation], priorities: dict[int, int], rules: Rules):
    metrics = objective_metrics(plan, priorities, rules)
    pair = (metrics["coverage_count"], round(metrics["observation_score"], 9))
    return pair if rules.objective == "coverage_then_score" else pair[::-1]


class _Plan:
    """Sparse sorted station and target indexes; no pairwise conflict matrix."""

    def __init__(self, observations: Iterable[Observation] = ()):
        self.stations: dict[str, list[Observation]] = defaultdict(list)
        self.starts: dict[str, list[float]] = defaultdict(list)
        self.targets: dict[int, list[Observation]] = defaultdict(list)
        self.free_gap_cache: dict = {}
        for observation in observations:
            self.add(observation)

    def add(self, observation: Observation):
        station = observation.station_id
        position = bisect_right(self.starts[station], observation.start)
        self.starts[station].insert(position, observation.start)
        self.stations[station].insert(position, observation)
        self.targets[observation.target_id].append(observation)
        self.free_gap_cache.pop(station, None)

    def remove(self, observation: Observation):
        station = observation.station_id
        position = self.stations[station].index(observation)
        self.stations[station].pop(position)
        self.starts[station].pop(position)
        self.targets[observation.target_id].remove(observation)
        self.free_gap_cache.pop(station, None)

    def free_gap_index(self, station: str, min_duration: float):
        """Index potentially usable raw gaps, rebuilt only after station changes.

        A raw gap shorter than the minimum duration is provably unusable even
        before adding nonnegative switching time. Skipping it is exact, and
        makes day-long windows cheap when thousands of tasks are packed tightly.
        """
        cached = self.free_gap_cache.get(station)
        if cached is not None and cached[0] == min_duration:
            return cached[1:]
        items = self.stations[station]
        left_edges, right_edges, pairs = [], [], []
        for position in range(len(items) + 1):
            previous = items[position - 1] if position else None
            following = items[position] if position < len(items) else None
            left = previous.end if previous is not None else -math.inf
            right = following.start if following is not None else math.inf
            if right - left + _EPS >= min_duration:
                left_edges.append(left)
                right_edges.append(right)
                pairs.append((previous, following))
        self.free_gap_cache[station] = (min_duration, left_edges, right_edges, pairs)
        return left_edges, right_edges, pairs

    def observations(self) -> list[Observation]:
        return sorted(
            (item for items in self.stations.values() for item in items),
            key=lambda item: (item.station_id, item.start, item.target_id, item.end),
        )


def _window_index(windows: Iterable[Window], priorities: dict[int, int]) -> dict:
    result: dict[int, list[Window]] = defaultdict(list)
    for window in windows:
        if not math.isfinite(window.start) or not math.isfinite(window.end):
            raise ValueError("Window endpoints must be finite")
        if window.end < window.start:
            raise ValueError("Window end precedes its start")
        if window.target_id not in priorities:
            raise ValueError(f"Missing priority for target {window.target_id}")
        result[window.target_id].append(window)
    for target, target_windows in result.items():
        target_windows.sort(key=lambda window: (window.start, window.end, window.station_id))
        # Deduplicate adjacent sorted windows without a million-entry tuple set.
        unique = []
        for window in target_windows:
            if not unique or window != unique[-1]:
                unique.append(window)
        result[target] = unique
    return dict(result)


def _repeat_segments(
    left: float, right: float, existing: list[Observation], rules: Rules
) -> list[tuple[float, float]]:
    """Subtract open forbidden start intervals; equality at the gap is legal."""
    if right + _EPS < left:
        return []
    if rules.repeat_gap_s is None or rules.repeat_gap_s == 0:
        return [(left, max(left, right))]
    segments = [(left, max(left, right))]
    gap = rules.repeat_gap_s
    for observation in existing:
        low, high = observation.start - gap, observation.start + gap
        updated = []
        for lower, upper in segments:
            if upper <= low or lower >= high:
                updated.append((lower, upper))
                continue
            if lower <= low:
                updated.append((lower, min(upper, low)))
            if upper >= high:
                updated.append((max(lower, high), upper))
        segments = updated
    return segments


def _gaps(plan: _Plan, window: Window, rules: Rules):
    """All free station gaps intersecting a physical window, found by bisect."""
    items = plan.stations[window.station_id]
    starts = plan.starts[window.station_id]
    position = bisect_right(starts, window.start) - 1
    intersecting_count = bisect_right(starts, window.end) - position
    if intersecting_count > 32:
        # A second sparse index avoids repeatedly walking every occupied task
        # inside a long window. Short windows keep the cheaper local sweep.
        left_edges, right_edges, pairs = plan.free_gap_index(
            window.station_id, rules.min_duration_s
        )
        gap_position = bisect_left(right_edges, window.start)
        while gap_position < len(pairs) and left_edges[gap_position] <= window.end:
            previous, following = pairs[gap_position]
            left, right = window.start, window.end
            if previous is not None:
                switch = rules.switch_s if previous.target_id != window.target_id else 0
                left = max(left, previous.end + switch)
            if following is not None:
                switch = rules.switch_s if following.target_id != window.target_id else 0
                right = min(right, following.start - switch)
            if right + _EPS >= left:
                yield left, max(left, right)
            gap_position += 1
        return
    previous = items[position] if position >= 0 else None
    next_position = position + 1
    while True:
        following = items[next_position] if next_position < len(items) else None
        left = window.start
        if previous is not None:
            switch = rules.switch_s if previous.target_id != window.target_id else 0
            left = max(left, previous.end + switch)
        right = window.end
        if following is not None:
            switch = rules.switch_s if following.target_id != window.target_id else 0
            right = min(right, following.start - switch)
        if right + _EPS >= left:
            yield left, max(left, right)
        if following is None or following.start > window.end:
            return
        previous = following
        next_position += 1


@dataclass(frozen=True)
class _Choice:
    observation: Observation
    cost: float


def _choices(
    target: int,
    index: dict[int, list[Window]],
    plan: _Plan,
    rules: Rules,
    horizon: float,
    duration: float | None = None,
) -> tuple[list[_Choice], int]:
    """Keep two best placements only AFTER inspecting every feasible gap."""
    if len(plan.targets[target]) >= rules.max_observations_per_target:
        return [], 0
    duration = rules.min_duration_s if duration is None else duration
    best: list[_Choice] = []
    alternatives = 0
    seen_gaps = set()
    for window in index[target]:
        if window.end - window.start + _EPS < duration:
            continue
        for left, right in _gaps(plan, window, rules):
            for lower, upper in _repeat_segments(
                left, right - duration, plan.targets[target], rules
            ):
                gap_key = (window.station_id, lower, upper)
                if gap_key in seen_gaps:
                    continue
                seen_gaps.add(gap_key)
                alternatives += 1
                endpoint_choices = []
                for start in {lower, upper}:
                    # Prefer placements leaving useful residual gaps, then earlier
                    # time. The cost ranks insertions, not physical feasibility.
                    before = max(0.0, start - left - rules.switch_s)
                    after = max(0.0, right - start - duration - rules.switch_s)
                    waste = sum(
                        value for value in (before, after)
                        if value + _EPS < rules.min_duration_s
                    )
                    cost = waste / rules.min_duration_s + 0.05 * start / horizon
                    endpoint_choices.append(
                        _Choice(
                            Observation(target, window.station_id, start, start + duration),
                            cost,
                        )
                    )
                candidate = min(
                    endpoint_choices,
                    key=lambda choice: (
                        choice.cost, choice.observation.start, choice.observation.station_id
                    ),
                )
                best.append(candidate)
                best.sort(key=lambda choice: (
                    choice.cost, choice.observation.start, choice.observation.station_id
                ))
                del best[2:]
    return best, alternatives


def _static_order(targets, index, priorities, mode, rng):
    tie = {target: rng.random() for target in targets}

    def key(target):
        windows = index[target]
        flexibility = sum(max(0.0, window.end - window.start) for window in windows)
        deadline = min(window.end for window in windows)
        if mode == "priority":
            return (-priorities[target], len(windows), flexibility, tie[target], target)
        if mode == "earliest":
            return (deadline, len(windows), -priorities[target], tie[target], target)
        return (len(windows), flexibility, -priorities[target], tie[target], target)

    return sorted(targets, key=key)


def _target_station_ranges(windows: list[Window]) -> dict:
    """A small interval index for precise regret-cache invalidation.

    Only the current regret batch is indexed. Sorted starts and prefix maximum
    ends answer whether ANY physical window can be affected by an insertion in
    O(log windows-per-target-per-station), including overlapping long windows.
    """
    grouped = defaultdict(list)
    for window in windows:
        grouped[window.station_id].append((window.start, window.end))
    result = {}
    for station, intervals in grouped.items():
        intervals.sort()
        starts, maximum_ends = [], []
        maximum = -math.inf
        for start, end in intervals:
            starts.append(start)
            maximum = max(maximum, end)
            maximum_ends.append(maximum)
        result[station] = (starts, maximum_ends)
    return result


def _insertion_affects_target(
    ranges: dict, observation: Observation, switch_s: float
) -> bool:
    """A non-overlap proves the cached choices still have identical gaps."""
    station_ranges = ranges.get(observation.station_id)
    if station_ranges is None:
        return False
    starts, maximum_ends = station_ranges
    position = bisect_right(starts, observation.end + switch_s) - 1
    return position >= 0 and maximum_ends[position] >= observation.start - switch_s


def _repair(
    index, priorities, rules, plan, mode, rng, horizon, deadline=None,
    regret_batch_size=32,
):
    """Repair every eligible target, with exact regret inside ordered batches.

    Batching controls ranking overhead only; every target and every window is
    eventually considered. This is a heuristic ordering, not exhaustive search.
    A deadline can stop repair early; the current feasible partial plan survives.
    """
    prefer_full = mode.startswith("full_")
    order_mode = mode.removeprefix("full_")
    pending = {
        target for target in index
        if len(plan.targets[target]) < rules.max_observations_per_target
    }
    while pending:
        ordered = _static_order(pending, index, priorities, order_mode, rng)
        changed = False
        if order_mode != "regret":
            for target in ordered:
                if deadline is not None and time.perf_counter() >= deadline:
                    return False
                choices, _ = _choices(
                    target, index, plan, rules, horizon,
                    duration=rules.full_duration_s if prefer_full else None,
                )
                if not choices and prefer_full:
                    choices, _ = _choices(target, index, plan, rules, horizon)
                if choices:
                    plan.add(choices[0].observation)
                    changed = True
        else:
            # Small instances use full regret; larger ones use declared batches.
            batch_size = len(ordered) if len(ordered) <= 96 else regret_batch_size
            for begin in range(0, len(ordered), batch_size):
                batch = set(ordered[begin:begin + batch_size])
                ranges = {
                    target: _target_station_ranges(index[target]) for target in batch
                }
                cached_choices = {}
                while batch:
                    if deadline is not None and time.perf_counter() >= deadline:
                        return False
                    candidates = []
                    infeasible = []
                    for target in sorted(batch):
                        if target not in cached_choices:
                            cached_choices[target] = _choices(
                                target, index, plan, rules, horizon
                            )
                        choices, count = cached_choices[target]
                        if not choices:
                            infeasible.append(target)
                            continue
                        regret = choices[1].cost - choices[0].cost if len(choices) > 1 else 1e6
                        key = (
                            count == 1, regret, -count, priorities[target], rng.random(), -target
                        )
                        candidates.append((key, target, choices[0]))
                    batch.difference_update(infeasible)
                    if not candidates:
                        break
                    _, target, choice = max(candidates, key=lambda item: item[0])
                    plan.add(choice.observation)
                    batch.remove(target)
                    for remaining in batch:
                        if _insertion_affects_target(
                            ranges[remaining], choice.observation, rules.switch_s
                        ):
                            cached_choices.pop(remaining, None)
                    changed = True
        if not changed or rules.max_observations_per_target == 1:
            return True
        pending = {
            target for target in pending
            if len(plan.targets[target]) < rules.max_observations_per_target
        }
    return True


def _extend(plan: _Plan, index, priorities, rules, deadline=None):
    """Increase durations without losing coverage or displacing other tasks."""
    order = sorted(
        plan.observations(),
        key=lambda item: (-priorities[item.target_id], item.start, item.station_id),
    )
    for observation in order:
        if deadline is not None and time.perf_counter() >= deadline:
            return False
        if observation.end - observation.start + _EPS >= rules.full_duration_s:
            continue
        plan.remove(observation)
        best = observation
        for window in index[observation.target_id]:
            if window.station_id != observation.station_id:
                continue
            if window.start > observation.start + _EPS or window.end + _EPS < observation.end:
                continue
            for left, right in _gaps(plan, window, rules):
                if left > observation.start + _EPS or right + _EPS < observation.end:
                    continue
                duration = min(rules.full_duration_s, right - left)
                if rules.score_mode == "step" and duration + _EPS < rules.full_duration_s:
                    continue
                lower = max(left, observation.end - duration)
                upper = min(observation.start, right - duration)
                segments = _repeat_segments(lower, upper, plan.targets[observation.target_id], rules)
                for lower_start, upper_start in segments:
                    start = min(max(observation.start, lower_start), upper_start)
                    candidate = Observation(
                        observation.target_id, observation.station_id, start, start + duration
                    )
                    if candidate.end - candidate.start > best.end - best.start + _EPS:
                        best = candidate
        plan.add(best)
    return True


def greedy_schedule(
    windows: list[Window],
    priorities: dict[int, int],
    rules: Rules,
    *,
    strategy: str = "portfolio",
    seed: int = 0,
    time_limit_s: float | None = None,
    diagnostics: dict | None = None,
) -> list[Observation]:
    """Return the best of scarcity, priority, deadline, and regret insertion.

    Coverage insertion uses minimum durations, followed by feasible extensions
    up to full-credit duration. A full-duration priority strategy also explores
    the score/coverage tradeoff. The portfolio chooses using ``rules.objective``.
    It is a baseline heuristic and does not claim an optimum for either objective.
    ``time_limit_s`` is a soft deadline checked between target/extension operations.
    The portfolio runs sequentially under one shared budget; it may stop before
    later strategies run. Pass a dictionary as ``diagnostics`` to record each
    strategy's actual runtime and completion, avoiding unfair budget accounting.
    """
    if strategy not in {
        "portfolio", "scarcity", "priority", "earliest", "regret",
        "full_priority", "full_scarcity",
    }:
        raise ValueError("Unknown greedy strategy")
    if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s < 0):
        raise ValueError("time_limit_s must be finite and nonnegative")
    started = time.perf_counter()
    deadline = None if time_limit_s is None else started + time_limit_s
    index = _window_index(windows, priorities)
    horizon = max(1.0, max((window.end for window in windows), default=1))
    strategies = (
        ("scarcity", "priority", "earliest", "regret", "full_priority")
        if strategy == "portfolio" else (strategy,)
    )
    best = []
    best_strategy = None
    strategy_runs = []
    for offset, mode in enumerate(strategies):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        plan = _Plan()
        strategy_started = time.perf_counter()
        construction_completed = _repair(
            index, priorities, rules, plan, mode, random.Random(seed + offset), horizon, deadline
        )
        extension_completed = _extend(plan, index, priorities, rules, deadline)
        candidate = plan.observations()
        strategy_runs.append({
            "strategy": mode,
            "elapsed_s": time.perf_counter() - strategy_started,
            "construction_completed": construction_completed,
            "extension_completed": extension_completed,
            "metrics": objective_metrics(candidate, priorities, rules),
        })
        if _objective(candidate, priorities, rules) > _objective(best, priorities, rules):
            best = candidate
            best_strategy = mode
    if diagnostics is not None:
        diagnostics.update({
            "algorithm": "greedy_portfolio" if strategy == "portfolio" else "greedy",
            "seed": seed,
            "requested_strategy": strategy,
            "time_limit_s": time_limit_s,
            "elapsed_s": time.perf_counter() - started,
            "budget_policy": "sequential strategies sharing one total soft deadline",
            "requested_strategy_count": len(strategies),
            "started_strategy_count": len(strategy_runs),
            "completed_strategy_count": sum(
                run["construction_completed"] and run["extension_completed"]
                for run in strategy_runs
            ),
            "best_strategy": best_strategy,
            "best_metrics": objective_metrics(best, priorities, rules),
            "strategy_runs": strategy_runs,
        })
    return best


def _validate_initial(plan, index, rules):
    """Reject malformed initial plans; final results still need external validation."""
    state = _Plan()
    for observation in sorted(plan, key=lambda item: (item.station_id, item.start)):
        if not math.isfinite(observation.start) or not math.isfinite(observation.end):
            raise ValueError("Initial plan contains nonfinite times")
        if observation.target_id not in index:
            raise ValueError("Initial plan target has no windows")
        if observation.end - observation.start + _EPS < rules.min_duration_s:
            raise ValueError("Initial plan observation is too short")
        if len(state.targets[observation.target_id]) >= rules.max_observations_per_target:
            raise ValueError("Initial plan exceeds the observation limit")
        fitting = False
        for window in index[observation.target_id]:
            if window.station_id != observation.station_id:
                continue
            for left, right in _gaps(state, window, rules):
                if left <= observation.start + _EPS and observation.end <= right + _EPS:
                    fitting = True
                    break
            if fitting:
                break
        if not fitting:
            raise ValueError("Initial plan violates a station gap or physical window")
        if rules.repeat_gap_s is not None and any(
            abs(observation.start - item.start) + _EPS < rules.repeat_gap_s
            for item in state.targets[observation.target_id]
        ):
            raise ValueError("Initial plan violates the repeat interval")
        state.add(observation)


def _destroy(plan, operator, count, rng, index, priorities, rules, horizon):
    if not plan:
        return []
    count = min(count, len(plan))
    if operator == "random":
        removed = rng.sample(plan, count)
    elif operator == "worst":
        # Low value per occupied second, slightly randomized to avoid cycling.
        removed = sorted(
            plan,
            key=lambda item: (
                observation_score(item, priorities[item.target_id], rules)
                / (item.end - item.start + rules.switch_s) * rng.uniform(0.8, 1.2)
            ),
        )[:count]
    else:
        seed = rng.choice(plan)
        seed_stations = {window.station_id for window in index[seed.target_id]}
        removed = sorted(
            plan,
            key=lambda item: (
                2.0 * (item.station_id != seed.station_id)
                + 0.5 * (item.station_id not in seed_stations)
                + abs(item.start - seed.start) / horizon
                + rng.random() * 0.15
            ),
        )[:count]
    removed_set = set(removed)
    return [item for item in plan if item not in removed_set]


def alns_schedule(
    windows: list[Window],
    priorities: dict[int, int],
    rules: Rules,
    *,
    initial: list[Observation] | None = None,
    iterations: int = 100,
    seed: int = 0,
    time_limit_s: float | None = None,
    destroy_fraction: tuple[float, float] = (0.1, 0.35),
    initial_temperature: float = 0.5,
    cooling: float = 0.995,
    reaction: float = 0.2,
) -> tuple[list[Observation], dict]:
    """Adaptive destroy/repair search with neutral and simulated-annealing moves.

    Destroy operators: random, spatial/temporal related, and low score density.
    Repair operators: regret insertion and randomized scarcity/priority order.
    Online exponential reward updates adapt both operator weight distributions.
    This is an adaptive classical baseline, not a trained interaction model.

    The soft total deadline includes initial construction if ``initial`` is
    absent. Iteration and target-operation boundaries are interruptible; one
    complete scan of a target's windows can exceed the deadline. All operator
    selections, acceptance decisions, incumbent quality, and elapsed times are
    recorded. An external independent validator must verify the returned plan.
    """
    if iterations < 0 or (time_limit_s is not None and time_limit_s < 0):
        raise ValueError("Budgets must be nonnegative")
    low, high = destroy_fraction
    if not (0 < low <= high <= 1):
        raise ValueError("Require 0 < destroy fraction bounds <= 1")
    if initial_temperature < 0 or not 0 < cooling <= 1 or not 0 < reaction <= 1:
        raise ValueError("Invalid adaptation or temperature parameter")
    started = time.perf_counter()
    deadline = None if time_limit_s is None else started + time_limit_s
    index = _window_index(windows, priorities)
    horizon = max(1.0, max((window.end for window in windows), default=1))
    if initial is None:
        remaining = None if deadline is None else max(0, deadline - time.perf_counter())
        current = greedy_schedule(
            windows, priorities, rules, seed=seed, time_limit_s=remaining
        )
    else:
        current = list(initial)
        _validate_initial(current, index, rules)
    best = list(current)
    current_key = _objective(current, priorities, rules)
    best_key = current_key
    rng = random.Random(seed)
    destroy_names = ("random", "related", "worst")
    repair_names = ("regret", "scarcity", "priority", "full_priority")
    destroy_weights = {name: 1.0 for name in destroy_names}
    repair_weights = {name: 1.0 for name in repair_names}
    uses = {name: 0 for name in destroy_names}
    repair_uses = {name: 0 for name in repair_names}
    # A normalized scalar controls stochastic acceptance only. Incumbent
    # improvement and best retention ALWAYS compare the lexicographic tuples.
    # In score-first/linear mode arbitrarily small score differences are
    # possible, so this scalar is not claimed to preserve lexicographic order.
    secondary_bound = (
        sum(priorities.values()) * rules.max_observations_per_target + 1
        if rules.objective == "coverage_then_score" else len(priorities) + 1
    )

    def energy(key):
        return key[0] + key[1] / secondary_bound

    trace = [{
        "iteration": 0,
        "elapsed_s": time.perf_counter() - started,
        "best": objective_metrics(best, priorities, rules),
        "current": objective_metrics(current, priorities, rules),
    }]
    accepted_count = 0
    neutral_count = 0
    completed = 0
    temperature = initial_temperature
    for iteration in range(1, iterations + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        destroy = rng.choices(destroy_names, [destroy_weights[name] for name in destroy_names])[0]
        repair = rng.choices(repair_names, [repair_weights[name] for name in repair_names])[0]
        uses[destroy] += 1
        repair_uses[repair] += 1
        count = max(1, math.ceil(len(current) * rng.uniform(low, high)))
        partial = _destroy(
            current, destroy, count, rng, index, priorities, rules, horizon
        )
        removed_count = len(current) - len(partial)
        candidate_state = _Plan(partial)
        _repair(index, priorities, rules, candidate_state, repair, rng, horizon, deadline)
        _extend(candidate_state, index, priorities, rules, deadline)
        candidate = candidate_state.observations()
        key = _objective(candidate, priorities, rules)
        delta = energy(key) - energy(current_key)
        accepted = key >= current_key or (
            temperature > 0 and rng.random() < math.exp(min(0.0, delta / temperature))
        )
        improved = key > best_key
        reward = 8.0 if improved else (3.0 if key > current_key else (1.0 if accepted else 0.0))
        if improved:
            best, best_key = candidate, key
        if accepted:
            neutral_count += key == current_key
            current, current_key = candidate, key
            accepted_count += 1
        destroy_weights[destroy] = max(
            0.1, (1 - reaction) * destroy_weights[destroy] + reaction * reward
        )
        repair_weights[repair] = max(
            0.1, (1 - reaction) * repair_weights[repair] + reaction * reward
        )
        completed = iteration
        trace.append({
            "iteration": iteration,
            "elapsed_s": time.perf_counter() - started,
            "destroy": destroy,
            "repair": repair,
            "removed_count": removed_count,
            "temperature": temperature,
            "accepted": accepted,
            "best_improved": improved,
            "best": objective_metrics(best, priorities, rules),
            "current": objective_metrics(current, priorities, rules),
        })
        temperature *= cooling
    elapsed = time.perf_counter() - started
    diagnostics = {
        "algorithm": "adaptive_large_neighbourhood_search",
        "seed": seed,
        "requested_iterations": iterations,
        "completed_iterations": completed,
        "time_limit_s": time_limit_s,
        "elapsed_s": elapsed,
        "budget_exhausted": deadline is not None and time.perf_counter() >= deadline,
        "accepted_moves": accepted_count,
        "neutral_accepted_moves": neutral_count,
        "destroy_weights": destroy_weights,
        "repair_weights": repair_weights,
        "destroy_uses": uses,
        "repair_uses": repair_uses,
        "initial_metrics": trace[0]["best"],
        "best_metrics": objective_metrics(best, priorities, rules),
        "regret_policy": "full pool up to 96 targets; ordered batches of 32 above",
        "candidate_policy": "all physical windows; continuous free-gap endpoints",
        "duration_policy": (
            "minimum-duration and full-duration-priority insertions; "
            "then coverage-preserving extension"
        ),
        "deadline_policy": "soft; checked between operations, not inside a target-window scan",
        "trace": trace,
    }
    return best, diagnostics
