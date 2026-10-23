"""伐谋候选放置规则的采纳检查。

    uv run python scripts/famou_adopt_check.py 候选.py [--full] [--tag 名称]

1. 子场景复评（沙箱内运行）：训练场景 2 个 + 伐谋看不到的留出场景 3 个，与人工规则比较（--cases v1/v2）；
2. --full：全量 H（300 次价格迭代 + 局部搜索 + 按秒精修），候选规则在本进程内运行——只对人工审阅过代码的候选用；
   输出计划 CSV 到 results/plan_H_evolved{_名称}.csv，之后用 `python -m obsplan validate` 做 Skyfield 校验。
结果写入 results/famou_adopt{_名称}.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from obsplan import famou_h_eval as fh  # noqa: E402
from obsplan.famou_eval import run_candidate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sub_ratios(source: bytes, cases) -> list[float]:
    """逐个场景在沙箱里运行（全目标场景每个上百万行，合在一起会超过输出文件大小上限）。"""
    out = []
    for c in cases:
        (u, v), = run_candidate(source, [c], names=fh.FEATURES, timeout_s=150, fsize=256 << 20)
        out.append(round(c.objective(np.asarray(u, float), np.asarray(v, float)) / c.base, 6))
    return out


def full_run(rule, label):
    from obsplan import lagrange
    from obsplan.config import Config
    from obsplan.data import load
    from obsplan.problem import Problem
    from obsplan.refine import finish
    from obsplan.schedule import evaluate
    from obsplan.visibility import get_windows
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    t = time.time()
    sch, info = lagrange.solve(pb, iters=300, verbose=False, rule=rule, name=label)
    t_plan = time.time() - t
    e = evaluate(cfg, ds, sch)
    t = time.time()
    ref, _ = finish(pb, sch)
    e2 = evaluate(cfg, ds, ref)
    row = {"grid": {k: e[k] for k in ("covered", "score", "combined", "tasks")},
           "refined": {k: e2[k] for k in ("covered", "score", "combined", "tasks")},
           "plan_seconds": round(t_plan, 1), "refine_seconds": round(time.time() - t, 1)}
    return row, cfg, ds, ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="全量时也跑一遍人工规则作对照")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cases", default="v2", help="v1（第一次实验，20% 目标 × 3 站）、v2（全目标 × 3–6 站）或 full（全量实例）")
    args = ap.parse_args()
    src = Path(args.program).read_bytes()
    out = {"program": str(args.program)}
    if args.cases == "full":                    # 全量评测（与第三次实验相同的打分）
        r = fh.evaluate_source(src, version="full", timeout_s=240)
        out["full_eval"] = {k: r.get(k) for k in ("ratio_vs_manual", "validity", "error_info")}
        print("全量评测（价格固定、快速局部搜索 + 精修）", out["full_eval"], flush=True)
    else:
        tr = fh.load_cases(version=args.cases, tag="train")
        ho = fh.load_cases(version=args.cases, tag="holdout")
        out["train"] = sub_ratios(src, tr)
        out["holdout"] = sub_ratios(src, ho)
        out["sub_mean"] = round(float(np.mean(out["train"] + out["holdout"])), 6)
        print("子场景（训练 / 留出）", out["train"], out["holdout"], "平均", out["sub_mean"], flush=True)
    if args.full:
        from obsplan.evolved import rule_from_file
        from obsplan.schedule import export_csv
        row, cfg, ds, ref = full_run(rule_from_file(args.program), "H 价格协调 + 伐谋放置规则")
        out["full"] = row
        print("全量", row, flush=True)
        if args.baseline:
            out["full_baseline"] = full_run(None, "H 价格协调")[0]
            print("全量人工规则", out["full_baseline"], flush=True)
        tag = f"_{args.tag}" if args.tag else ""
        csv = ROOT / "results" / f"plan_H_evolved{tag}.csv"
        export_csv(cfg, ds, ref, csv)
        out["plan_csv"] = f"results/{csv.name}"
    tag = f"_{args.tag}" if args.tag else ""
    (ROOT / "results" / f"famou_adopt{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
