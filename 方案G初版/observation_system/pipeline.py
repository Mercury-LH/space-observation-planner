from __future__ import annotations

import hashlib
import json
import time
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from scheme_g.models import CandidateTask, PlanResult

from .candidate_service import build_candidates
from .config import SystemConfig
from .data_loader import DataAudit, load_competition_data
from .domain import GroundStation, ObservationWindow, Target
from .planners import create_planner
from .reporting import export_pipeline_result
from .visibility_engine import SkyfieldVisibilityEngine
from .window_io import read_windows, write_windows


@dataclass(slots=True)
class PipelineResult:
    plan: PlanResult
    windows: list[ObservationWindow]
    candidates: list[CandidateTask]
    audit: DataAudit
    visibility_failures: list[str]
    elapsed_seconds: dict[str, float]


class ObservationPlanningPipeline:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.output_dir = Path(config.output.directory)

    def load_data(self) -> tuple[list[Target], list[GroundStation], DataAudit]:
        targets, stations, audit = load_competition_data(
            self.config.data.tle_path,
            self.config.data.station_path,
            self.config.data.priority_dir,
        )
        if not audit.valid:
            raise ValueError(
                "数据审计失败: "
                f"缺少优先级={len(audit.missing_priority_ids)}, "
                f"优先级无轨道={len(audit.priority_ids_without_tle)}, "
                f"重复目标={len(audit.duplicate_target_ids)}"
            )
        return targets, stations, audit

    def _active_data(
        self,
        targets: list[Target],
        stations: list[GroundStation],
    ) -> tuple[list[Target], list[GroundStation]]:
        if self.config.visibility.target_limit:
            targets = targets[: self.config.visibility.target_limit]
        if self.config.visibility.station_limit:
            stations = stations[: self.config.visibility.station_limit]
        return targets, stations

    def _adjust_limited_normalizers(self, targets: list[Target]) -> None:
        """小规模验证时按实际目标集缩放两项指标；正式全量配置不变。"""
        active_targets, _ = self._active_data(targets, [])
        if self.config.visibility.target_limit:
            self.config.planner.coverage_normalizer = max(len(active_targets), 1)
            self.config.planner.score_normalizer = max(
                3.0 * sum(target.base_score for target in active_targets),
                1.0,
            )

    def _window_fingerprint(
        self,
        targets: list[Target],
        stations: list[GroundStation],
    ) -> str:
        payload = {
            "target_ids": [target.target_id for target in targets],
            "station_ids": [station.station_id for station in stations],
            "scenario": {
                **asdict(self.config.scenario),
                "start_time": self.config.scenario.start_time.isoformat(),
                "end_time": self.config.scenario.end_time.isoformat(),
            },
            "visibility": asdict(self.config.visibility),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compute_windows(
        self,
        targets: list[Target],
        stations: list[GroundStation],
        progress: Callable[[int, int], None] | None = None,
        force: bool = False,
    ) -> tuple[list[ObservationWindow], list[str]]:
        active_targets, active_stations = self._active_data(targets, stations)
        windows_path = self.output_dir / "windows.csv"
        metadata_path = self.output_dir / "windows.meta.json"
        fingerprint = self._window_fingerprint(active_targets, active_stations)

        if (
            self.config.output.cache_windows
            and not force
            and windows_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint:
                return read_windows(windows_path), metadata.get("failures", [])

        engine = SkyfieldVisibilityEngine(self.config.scenario, self.config.visibility)
        result = engine.compute(active_targets, active_stations, progress=progress)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_windows(windows_path, result.windows)
        metadata_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "target_count": len(active_targets),
                    "station_count": len(active_stations),
                    "window_count": len(result.windows),
                    "failures": result.failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return result.windows, result.failures

    def run(
        self,
        progress: Callable[[int, int], None] | None = None,
        force_windows: bool = False,
    ) -> PipelineResult:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        targets, stations, audit = self.load_data()
        self._adjust_limited_normalizers(targets)
        timings["load_data"] = time.perf_counter() - started

        started = time.perf_counter()
        windows, failures = self.compute_windows(
            targets,
            stations,
            progress=progress,
            force=force_windows,
        )
        timings["compute_windows"] = time.perf_counter() - started

        started = time.perf_counter()
        candidates = build_candidates(windows, self.config.candidates)
        timings["build_candidates"] = time.perf_counter() - started
        if not candidates:
            raise RuntimeError("未生成任何候选任务，请检查时间范围、站点和可见性规则")

        started = time.perf_counter()
        planner = create_planner(self.config.planner_name, self.config.planner)
        plan = planner.plan(candidates)
        timings["planning"] = time.perf_counter() - started

        if plan.validation_errors:
            raise RuntimeError(
                "规划结果未通过独立校验:\n" + "\n".join(plan.validation_errors[:20])
            )

        timings["total"] = sum(timings.values())
        export_pipeline_result(
            self.output_dir,
            plan,
            audit,
            windows,
            len(candidates),
            failures,
            timings,
            active_target_count=len(self._active_data(targets, stations)[0]),
            active_station_count=len(self._active_data(targets, stations)[1]),
        )
        return PipelineResult(
            plan=plan,
            windows=windows,
            candidates=candidates,
            audit=audit,
            visibility_failures=failures,
            elapsed_seconds=timings,
        )

    def compare_planners(
        self,
        planner_names: list[str],
        progress: Callable[[int, int], None] | None = None,
        force_windows: bool = False,
    ) -> list[dict[str, object]]:
        targets, stations, audit = self.load_data()
        self._adjust_limited_normalizers(targets)
        windows, failures = self.compute_windows(
            targets,
            stations,
            progress=progress,
            force=force_windows,
        )
        candidates = build_candidates(windows, self.config.candidates)
        if not candidates:
            raise RuntimeError("未生成任何候选任务")

        rows: list[dict[str, object]] = []
        comparison_root = self.output_dir / "planner_comparison"
        active_targets, active_stations = self._active_data(targets, stations)
        for planner_name in planner_names:
            started = time.perf_counter()
            planner = create_planner(planner_name, self.config.planner)
            result = planner.plan(candidates)
            elapsed = time.perf_counter() - started
            if result.validation_errors:
                raise RuntimeError(
                    f"规划器{planner_name}输出不合法:\n"
                    + "\n".join(result.validation_errors[:20])
                )
            export_pipeline_result(
                comparison_root / planner_name,
                result,
                audit,
                windows,
                len(candidates),
                failures,
                {"planning": elapsed, "total": elapsed},
                active_target_count=len(active_targets),
                active_station_count=len(active_stations),
            )
            rows.append(
                {
                    "planner": planner_name,
                    "algorithm": result.algorithm,
                    "task_count": result.metrics.task_count,
                    "covered_targets": result.metrics.covered_targets,
                    "total_score": result.metrics.total_score,
                    "composite_score": result.metrics.composite_score,
                    "planning_seconds": elapsed,
                    "valid": not result.validation_errors,
                }
            )

        comparison_root.mkdir(parents=True, exist_ok=True)
        with (comparison_root / "comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows
