"""按秒大邻域搜索（obsplan/second_lns.py）：从一份按秒计划出发，逐站逐段用 CP-SAT 按秒重排，变好才接受。

    uv run python scripts/second_lns.py --plan outputs/long_run/best_plan.csv --max-slices 20     # 试几段
    uv run python scripts/second_lns.py --plan ... --passes 2 --workers 8                          # 全部站、全部夜晚

输出（--out，默认 outputs/second_lns/）：best_plan.csv、best.json、log.jsonl（每段一行）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.second_lns import Context, slices, solve_slice  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "second_lns"))
    ap.add_argument("--hours", type=float, default=2.0, help="每段长度（小时）")
    ap.add_argument("--step", type=float, default=1.0, help="段与段的步长（小时）")
    ap.add_argument("--limit", type=float, default=10.0, help="每段 CP-SAT 时限（秒）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--hint-keep", type=float, default=0.5, help="提示里随机保留的段内任务比例（拆掉其余的再重建）")
    ap.add_argument("--stations", default="", help="只做这些站（逗号分隔，0 起），默认全部")
    ap.add_argument("--max-slices", type=int, default=0, help="试验用：最多做多少段（均匀抽取）")
    ap.add_argument("--max-hours", type=float, default=0.0, help="总时长上限（小时），0 = 不限")
    args = ap.parse_args()
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    sch, skipped = read_plan(cfg, ds, Path(args.plan))
    assert not skipped, skipped[:3]
    sch.start_s = np.rint(sch.start_s).astype(np.int64)
    sch.dur_s = np.rint(sch.dur_s).astype(np.int64)
    e0 = evaluate(cfg, ds, sch)
    print(f"起点 {Path(args.plan).name}：综合 {e0['combined']:.6f}，覆盖 {e0['covered']}，任务 {e0['tasks']}", flush=True)
    ctx = Context.build(pb)
    stations = [int(x) for x in args.stations.split(",") if x] or None
    jobs = slices(pb, args.hours, args.step, stations)
    if args.max_slices:
        jobs = [jobs[i] for i in np.linspace(0, len(jobs) - 1, args.max_slices).round().astype(int)]
    print(f"{len(jobs)} 段 × {args.passes} 遍，每段 {args.hours:g} 小时、时限 {args.limit:g} 秒", flush=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    best = e0["combined"]
    for p in range(args.passes):
        acc = 0
        for k, (s, A, B) in enumerate(jobs):
            if args.max_hours and time.time() - t_start > args.max_hours * 3600:
                break
            sch, info = solve_slice(ctx, sch, s, A, B, time_limit=args.limit, workers=args.workers, seed=p * 1000 + k,
                                     hint_keep=args.hint_keep)
            info["pass"] = p
            log.write(json.dumps(info, ensure_ascii=False, default=float) + "\n")
            log.flush()
            if info.get("accepted"):
                acc += 1
                best = info["combined_after"]
            if info.get("accepted") or k % 20 == 0:
                print(f"  第 {p + 1} 遍 {k + 1}/{len(jobs)} 站 {s} {A / 3600:.1f}–{B / 3600:.1f} h：候选 "
                      f"{info.get('candidates', 0)}，{info.get('status', info.get('skipped', ''))}，"
                      f"{'接受' if info.get('accepted') else '不变'}，当前 {best:.6f}"
                      f"（{time.time() - t_start:.0f} 秒）", flush=True)
        print(f"第 {p + 1} 遍：接受 {acc} 段，综合 {best:.6f}", flush=True)
        if not acc:
            break
    e1 = evaluate(cfg, ds, sch)
    bad = structural_violations(pb, sch, by_second=True, extra_windows=ctx.extra_windows())
    res = {"start": {k: e0[k] for k in ("combined", "covered", "score", "tasks")},
           "final": {k: e1[k] for k in ("combined", "covered", "score", "tasks")},
           "violations": {k: len(v) for k, v in bad.items()}, "seconds": round(time.time() - t_start),
           "args": vars(args)}
    (out / "best.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    sch.algo = "长跑 + 按秒大邻域搜索"
    export_csv(cfg, ds, sch, out / "best_plan.csv")
    print(json.dumps(res, ensure_ascii=False, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
