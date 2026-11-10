"""算法效率测速：默认流水线各阶段用时，并给出计划指纹，用来核对提速前后结果逐位相同。

    uv run python scripts/bench_pipeline.py --mode fast            # 快速模式
    uv run python scripts/bench_pipeline.py --mode quality --profile  # 高质量模式，并打印 cProfile 热点

各阶段用包装函数计时，不改求解器本身。嵌套关系：可行解恢复 ⊃ 伐谋规则特征 + 伐谋规则打分 + 恢复贪心。
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import famou_h_eval, lagrange  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.evolved import load_rule  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

T: dict[str, float] = {}


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[name] = T.get(name, 0.0) + time.perf_counter() - t
    return wrapper


def digest(sch) -> str:
    h = hashlib.sha256()
    for arr in (sch.station, sch.target, sch.start_s, sch.dur_s):
        h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "quality"], default="fast")
    ap.add_argument("--manual-rule", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", help="把结果写成 JSON")
    ap.add_argument("--save", help="把最终计划存成 npz（逐位比对用）")
    args = ap.parse_args()

    lagrange._subproblem = timed("价格迭代子问题", lagrange._subproblem)
    lagrange.candidates = timed("候选生成", lagrange.candidates)
    lagrange.recover = timed("可行解恢复", lagrange.recover)
    lagrange.run_greedy = timed("恢复贪心", lagrange.run_greedy)
    famou_h_eval.feature_rows = timed("伐谋规则特征", famou_h_eval.feature_rows)
    lagrange.improve = timed("局部搜索", lagrange.improve)

    cfg = Config()
    t0 = time.perf_counter()
    ds = load(cfg)
    t1 = time.perf_counter()
    win = get_windows(cfg, ds)
    t2 = time.perf_counter()
    pb = Problem.build(cfg, ds, win)
    t3 = time.perf_counter()
    rule = None if args.manual_rule else load_rule()
    if rule is not None:
        rule = timed("伐谋规则打分", rule)
    fast = args.mode == "fast"
    prof = cProfile.Profile() if args.profile else None
    if prof:
        prof.enable()
    sch, info = lagrange.solve(pb, iters=150 if fast else 300, fast=fast, verbose=False, rule=rule)
    t4 = time.perf_counter()
    grid_digest = digest(sch)
    sch, rinfo = finish(pb, sch)
    t5 = time.perf_counter()
    if prof:
        prof.disable()
    ev = evaluate(cfg, ds, sch)
    res = {
        "mode": args.mode, "rule": "manual" if rule is None else "famou",
        "seconds": {"数据读取": round(t1 - t0, 2), "可见窗口（缓存）": round(t2 - t1, 2), "建模": round(t3 - t2, 2),
                    "价格协调与局部搜索": round(t4 - t3, 2), "按秒精修": round(t5 - t4, 2),
                    "规划合计": round(t5 - t3, 2), "端到端": round(t5 - t0, 2)},
        "phases": {k: round(v, 2) for k, v in T.items()},
        "local_search": {k: round(v, 2) for k, v in info["ls"].get("time", {}).items()},
        "refine": rinfo,
        "metrics": {"covered": ev["covered"], "score": round(ev["score"], 3), "combined": round(ev["combined"], 6),
                    "tasks": ev["tasks"]},
        "digest_grid": grid_digest, "digest_final": digest(sch),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(25)
        print(s.getvalue()[:6000])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, station=sch.station, target=sch.target, start_s=sch.start_s, dur_s=sch.dur_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
