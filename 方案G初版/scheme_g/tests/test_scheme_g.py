from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scheme_g.conflicts import build_station_conflicts, build_target_conflicts
from scheme_g.candidate_builder import VisibilityWindow, build_candidates_from_windows
from scheme_g.demo import generate_demo_candidates
from scheme_g.evaluator import evaluate_schedule
from scheme_g.features import build_features
from scheme_g.filtering import safe_filter_candidates
from scheme_g.greedy import greedy_schedule
from scheme_g.io import load_candidates
from scheme_g.models import CandidateTask, PlannerConfig, calculate_task_score
from scheme_g.planner import SchemeGPlanner
from scheme_g.scorer import LinearTaskScorer
from scheme_g.validator import validate_schedule


UTC = timezone.utc


def make_task(
    task_id: str,
    station: str,
    target: str,
    start: datetime,
    duration: int = 120,
    window_id: str | None = None,
) -> CandidateTask:
    end = start + timedelta(seconds=duration)
    base_score = 5.0
    return CandidateTask(
        task_id=task_id,
        window_id=window_id or task_id,
        station_id=station,
        target_id=target,
        priority="HIGH",
        window_start=start,
        window_end=end,
        start=start,
        end=end,
        base_score=base_score,
        task_score=calculate_task_score(base_score, duration),
        duration_seconds=duration,
        min_elevation=12.0,
        max_elevation=70.0,
        max_sun_elevation=-8.0,
    )


class SchemeGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlannerConfig(ai_weights={})
        self.start = datetime(2026, 10, 1, tzinfo=UTC)

    def test_score_rule(self) -> None:
        self.assertAlmostEqual(calculate_task_score(5.0, 120), 3.0)
        self.assertAlmostEqual(calculate_task_score(5.0, 180), 4.0)
        self.assertAlmostEqual(calculate_task_score(5.0, 240), 5.0)

    def test_station_switch_conflict(self) -> None:
        first = make_task("1", "S1", "T1", self.start)
        second = make_task("2", "S1", "T2", self.start + timedelta(seconds=140))
        third = make_task("3", "S1", "T3", self.start + timedelta(seconds=150))
        conflicts = set(build_station_conflicts([first, second, third], self.config))
        self.assertIn(("1", "2"), conflicts)
        self.assertNotIn(("1", "3"), conflicts)

    def test_target_cooldown_conflict(self) -> None:
        first = make_task("1", "S1", "T1", self.start)
        second = make_task("2", "S2", "T1", self.start + timedelta(hours=47))
        third = make_task(
            "3",
            "S2",
            "T1",
            first.end + timedelta(hours=48),
        )
        conflicts = set(build_target_conflicts([first, second, third], self.config))
        self.assertIn(("1", "2"), conflicts)
        self.assertNotIn(("1", "3"), conflicts)

    def test_filter_keeps_each_target(self) -> None:
        tasks = [
            make_task(str(index), "S1", f"T{index}", self.start + timedelta(minutes=index * 5))
            for index in range(1, 8)
        ]
        build_features(tasks, self.config)
        LinearTaskScorer().score_tasks(tasks)
        filtered = safe_filter_candidates(tasks, self.config)
        self.assertEqual({task.target_id for task in filtered}, {task.target_id for task in tasks})

    def test_greedy_is_valid(self) -> None:
        tasks = [
            make_task("1", "S1", "T1", self.start),
            make_task("2", "S1", "T2", self.start + timedelta(seconds=60)),
            make_task("3", "S1", "T3", self.start + timedelta(seconds=150)),
        ]
        build_features(tasks, self.config)
        LinearTaskScorer().score_tasks(tasks)
        schedule = greedy_schedule(tasks, self.config)
        validation = validate_schedule(schedule, self.config)
        self.assertTrue(validation.valid, validation.errors)

    def test_demo_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.csv"
            generate_demo_candidates(path, target_count=5, station_count=2)
            tasks = load_candidates(path)
            self.assertGreater(len(tasks), 0)
            self.assertGreater(evaluate_schedule([], self.config).composite_score, -1)

    def test_scheme_g_end_to_end(self) -> None:
        tasks = [
            make_task("1", "S1", "T1", self.start),
            make_task("2", "S1", "T2", self.start + timedelta(seconds=60)),
            make_task("3", "S1", "T3", self.start + timedelta(seconds=150)),
            make_task("4", "S2", "T2", self.start + timedelta(seconds=60)),
        ]
        config = PlannerConfig(
            ai_weights={},
            coverage_normalizer=3,
            score_normalizer=15.0,
            solver_time_limit_seconds=5,
            local_search_max_removals=5,
        )
        result = SchemeGPlanner(config).plan(tasks)
        self.assertFalse(result.validation_errors)
        self.assertEqual(result.metrics.covered_targets, 3)

    def test_window_adapter(self) -> None:
        window = VisibilityWindow(
            window_id="W1",
            station_id="S1",
            target_id="T1",
            priority="高",
            start=self.start + timedelta(seconds=7),
            end=self.start + timedelta(minutes=10),
            min_elevation=12.0,
            max_elevation=70.0,
            max_sun_elevation=-8.0,
        )
        tasks = build_candidates_from_windows(
            [window],
            durations_seconds=(120, 240),
            grid_seconds=30,
            max_starts_per_window_duration=4,
        )
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task.start.second in {0, 30} for task in tasks))
        self.assertTrue(all(task.end <= window.end for task in tasks))


if __name__ == "__main__":
    unittest.main()
