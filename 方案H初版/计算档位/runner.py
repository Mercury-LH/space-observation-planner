"""六档文件夹的共同命令行入口；只装配配置，正式评测交给 benchmark。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

SCHEME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCHEME_DIR))

from obsplan import benchmark  # noqa: E402
from obsplan.effort import get_effort, spec_for_effort  # noqa: E402


def main(level: str, argv=None) -> int:
    profile = get_effort(level)
    parser = argparse.ArgumentParser(description=f"{profile.title}：{profile.description}")
    parser.add_argument("--show-config", action="store_true", help="仅显示解析后的评测配置，不求解或创建输出目录")
    parser.add_argument("--output", help="新的输出目录，相对方案H初版；必须在 outputs/ 内")
    parser.add_argument("--repeats", type=int, default=1, help="重复次数；最终候选建议至少 2 次")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026, help="首轮随机种子；每次重复加 1")
    parser.add_argument("--timeout-s", type=float, default=3600 if level == "ultra" else 1200,
                        help="每次运行的进程超时秒数，包含准备、求解与校验")
    parser.add_argument("--time-weight", type=float, default=benchmark.DEFAULT_TIME_WEIGHT)
    parser.add_argument("--time-scale-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    try:
        spec = spec_for_effort(level, seed=args.seed, workers=args.workers, full_problem=True)
        if args.repeats < 1 or args.seed + args.repeats - 1 > 2**31 - 1:
            raise ValueError("repeats 必须为正整数，且最后一轮 seed 不得超过 2**31-1")
        if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
            raise ValueError("timeout-s 必须是有限正数")
        benchmark.utility(0, 0, args.time_weight, args.time_scale_s)
    except ValueError as exc:
        parser.error(str(exc))

    name = profile.benchmark_name
    suite = {
        "schema_version": 1, "effort_level": level,
        "presets": [name], "repeats": args.repeats,
        "workers": args.workers, "timeout_s": args.timeout_s,
        "time_weight": args.time_weight, "time_scale_s": args.time_scale_s,
        "weights": sorted(set((*benchmark.WEIGHTS, args.time_weight))),
        "config": {"start_utc": "2026-08-01T00:00:00", "days": 6.0, "seed": args.seed},
        "solvers": {name: {"entrypoint": "obsplan.portfolio:solve", "spec": spec}},
    }
    encoded = json.dumps(suite, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.show_config:
        print(encoded, end="")
        return 0
    if not args.output:
        parser.error("实际运行必须指定 --output 新目录；仅查看配置请使用 --show-config")
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = SCHEME_DIR / output
    output = output.resolve()
    if not output.is_relative_to((SCHEME_DIR / "outputs").resolve()):
        parser.error("输出目录必须位于方案H初版/outputs/ 内")
    if output.exists():
        parser.error("输出目录已存在，请使用新的目录以保留历史运行")

    with TemporaryDirectory(prefix=f"obsplan-{level}-") as temporary:
        source = Path(temporary) / "suite.json"
        source.write_text(encoded, encoding="utf-8")
        # No alternate solver, validation bypass or cached plan is introduced.
        code = benchmark.main(["run", "--suite", str(source), "--output", str(output)])
        if output.is_dir():
            (output / "source-suite.json").write_text(encoded, encoding="utf-8")
        return code
