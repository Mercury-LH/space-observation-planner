from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_system_config
from .data_loader import load_competition_data
from .pipeline import ObservationPlanningPipeline
from .planners import registered_planners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="空间目标自主观测任务规划系统")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser("inspect", help="读取并审计比赛数据")
    inspect.add_argument("--config", required=True)

    windows = subcommands.add_parser("windows", help="仅计算合法可见窗口")
    windows.add_argument("--config", required=True)
    windows.add_argument("--limit-targets", type=int)
    windows.add_argument("--limit-stations", type=int)
    windows.add_argument("--output-dir")
    windows.add_argument("--force", action="store_true")

    run = subcommands.add_parser("run", help="运行完整规划流程")
    run.add_argument("--config", required=True)
    run.add_argument("--planner", choices=registered_planners())
    run.add_argument("--limit-targets", type=int)
    run.add_argument("--limit-stations", type=int)
    run.add_argument("--output-dir")
    run.add_argument("--force-windows", action="store_true")

    compare = subcommands.add_parser("compare", help="在同一窗口和候选集上比较规划器")
    compare.add_argument("--config", required=True)
    compare.add_argument(
        "--planners",
        nargs="+",
        choices=registered_planners(),
        default=registered_planners(),
    )
    compare.add_argument("--limit-targets", type=int)
    compare.add_argument("--limit-stations", type=int)
    compare.add_argument("--output-dir")
    compare.add_argument("--force-windows", action="store_true")

    subcommands.add_parser("planners", help="列出可用规划器")
    return parser


def apply_overrides(config, args) -> None:
    if getattr(args, "limit_targets", None) is not None:
        config.visibility.target_limit = args.limit_targets
        config.planner.coverage_normalizer = args.limit_targets
    if getattr(args, "limit_stations", None) is not None:
        config.visibility.station_limit = args.limit_stations
    if getattr(args, "planner", None):
        config.planner_name = args.planner
    if getattr(args, "output_dir", None):
        config.output.directory = args.output_dir


def progress_printer(processed: int, total: int) -> None:
    if processed == total or processed == 1 or processed % max(total // 20, 1) == 0:
        print(f"可见窗口计算: {processed}/{total} ({processed / total:.1%})")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "planners":
        print("\n".join(registered_planners()))
        return

    config = load_system_config(args.config)
    apply_overrides(config, args)

    if args.command == "inspect":
        _, _, audit = load_competition_data(
            config.data.tle_path,
            config.data.station_path,
            config.data.priority_dir,
        )
        print(
            json.dumps(
                {
                    "target_count": audit.target_count,
                    "station_count": audit.station_count,
                    "priority_counts": audit.priority_counts,
                    "missing_priority_count": len(audit.missing_priority_ids),
                    "priority_without_tle_count": len(audit.priority_ids_without_tle),
                    "duplicate_target_count": len(audit.duplicate_target_ids),
                    "valid": audit.valid,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    pipeline = ObservationPlanningPipeline(config)
    if args.command == "windows":
        targets, stations, audit = pipeline.load_data()
        windows, failures = pipeline.compute_windows(
            targets,
            stations,
            progress=progress_printer,
            force=args.force,
        )
        print(
            json.dumps(
                {
                    "window_count": len(windows),
                    "failure_count": len(failures),
                    "output": str(Path(config.output.directory) / "windows.csv"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "compare":
        rows = pipeline.compare_planners(
            args.planners,
            progress=progress_printer,
            force_windows=args.force_windows,
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    result = pipeline.run(
        progress=progress_printer,
        force_windows=args.force_windows,
    )
    print(
        json.dumps(
            {
                "planner": result.plan.algorithm,
                **result.plan.metrics.to_dict(),
                "windows": len(result.windows),
                "candidates": len(result.candidates),
                "visibility_failures": len(result.visibility_failures),
                "output": config.output.directory,
                "timing_seconds": result.elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
