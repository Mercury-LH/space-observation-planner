"""Enumerate a synthetic coverage-only example; this is not a contest benchmark.

All windows and candidates are constructed, not propagated from the supplied TLEs.
Time is in seconds after 09:00 UTC. Observations last 120 seconds and switching
between different targets on one station takes at least 30 seconds. Repeated
observations are allowed, but only distinct-target coverage is evaluated.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
import json


@dataclass(frozen=True)
class Candidate:
    name: str
    target: str
    station: str
    start: int
    end: int


CANDIDATES = (
    Candidate("A_on_S1", "A", "S1", 0, 120),
    Candidate("A_on_S2", "A", "S2", 0, 120),
    Candidate("B_early", "B", "S2", 0, 120),
    Candidate("B_late", "B", "S2", 150, 270),
    Candidate("C_on_S1", "C", "S1", 0, 120),
)
INITIAL = (CANDIDATES[0], CANDIDATES[2])


def feasible(plan: tuple[Candidate, ...]) -> bool:
    for station in ("S1", "S2"):
        ordered = sorted((c for c in plan if c.station == station), key=lambda c: c.start)
        for first, second in zip(ordered, ordered[1:]):
            gap = 30 if first.target != second.target else 0
            if second.start < first.end + gap:
                return False
    return True


def coverage(plan: tuple[Candidate, ...]) -> int:
    return len({c.target for c in plan})


def optimize(unlocked: frozenset[str]) -> dict:
    fixed = tuple(c for c in INITIAL if c.station not in unlocked)
    eligible = tuple(c for c in CANDIDATES if c.station in unlocked)
    best = INITIAL
    feasible_plans = 0
    for size in range(len(eligible) + 1):
        for subset in combinations(eligible, size):
            plan = fixed + subset
            if feasible(plan):
                feasible_plans += 1
                if (coverage(plan), -len(plan)) > (coverage(best), -len(best)):
                    best = plan
    return {
        "unlocked_stations": sorted(unlocked),
        "coverage": coverage(best),
        "gain": coverage(best) - coverage(INITIAL),
        "feasible_candidate_subsets": feasible_plans,
        "plan": [asdict(c) for c in best],
    }


def main() -> None:
    single_one = optimize(frozenset({"S1"}))
    single_two = optimize(frozenset({"S2"}))
    joint = optimize(frozenset({"S1", "S2"}))
    residual = joint["gain"] - single_one["gain"] - single_two["gain"]
    assert feasible(INITIAL)
    assert [single_one["gain"], single_two["gain"], joint["gain"], residual] == [0, 0, 1, 1]
    print(json.dumps({
        "scope": "synthetic exact enumeration; coverage only; no learned model",
        "initial_coverage": coverage(INITIAL),
        "repairs": [single_one, single_two, joint],
        "joint_interaction_residual": residual,
        "limitation": "Neutral moves or classical exchange chains can also solve this example.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
