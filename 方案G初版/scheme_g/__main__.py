from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_algorithms
from .demo import generate_demo_candidates
from .evolution import evolve_weights
from .io import load_candidates, load_config, write_json, write_metrics, write_schedule
from .planner import SchemeGPlanner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="方案G AI筛选加CP-SAT调度器")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="生成演示候选任务")
    demo.add_argument("--output", required=True)
    demo.add_argument("--targets", type=int, default=40)
    demo.add_argument("--stations", type=int, default=3)

    plan = commands.add_parser("plan", help="运行方案G")
    plan.add_argument("--candidates", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)

    compare = commands.add_parser("compare", help="比较基线与方案G")
    compare.add_argument("--candidates", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--output", required=True)

    tune = commands.add_parser("tune", help="遗传搜索AI评分权重")
    tune.add_argument("--candidates", required=True)
    tune.add_argument("--config", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--population", type=int, default=24)
    tune.add_argument("--generations", type=int, default=20)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "demo":
        path = generate_demo_candidates(
            args.output,
            target_count=args.targets,
            station_count=args.stations,
        )
        print(f"演示数据已生成: {path}")
        return

    config = load_config(args.config)
    tasks = load_candidates(args.candidates)

    if args.command == "plan":
        output_dir = Path(args.output)
        result = SchemeGPlanner(config).plan(tasks)
        write_schedule(output_dir / "schedule.csv", result.selected_tasks)
        write_metrics(
            output_dir / "metrics.json",
            result.metrics,
            {
                "algorithm": result.algorithm,
                "solver_status": result.solver_status,
                "solver_wall_time": result.solver_wall_time,
                "best_bound": result.best_bound,
                "candidate_count": result.candidate_count,
                "filtered_candidate_count": result.filtered_candidate_count,
                "validation_errors": result.validation_errors,
            },
        )
        print(json.dumps(result.metrics.to_dict(), ensure_ascii=False, indent=2))
        if result.validation_errors:
            raise SystemExit("输出未通过校验，请查看metrics.json")
        return

    if args.command == "compare":
        rows = compare_algorithms(tasks, config, args.output)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.command == "tune":
        weights, history = evolve_weights(
            tasks,
            config,
            population_size=args.population,
            generations=args.generations,
        )
        write_json(args.output, weights)
        history_path = Path(args.output).with_name(Path(args.output).stem + "_history.json")
        write_json(history_path, history)
        print(json.dumps(weights, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

