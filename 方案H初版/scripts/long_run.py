"""长跑优化（随时可停）：在当前最好计划上反复"生成一份新的 H+ 计划 → 解合并（CP-SAT 分块）→ 局部搜索"，
只接受变好的，每次变好都把最好计划存盘。用来回答"多给时间能不能更好"，也给比赛提交准备最好的计划。

    uv run python scripts/long_run.py --hours 3              # 从 results/plan_{G+,E,H+}.csv 出发
    uv run python scripts/long_run.py --hours 3 --resume     # 从上次存的最好计划接着跑

每轮的新计划来自不同的设置（价格迭代次数、恢复时机、快速/完整局部搜索、伐谋/人工放置规则），
不改求解器本身；合并池 = 当前最好 + 固定参照（G+、E）+ 最近几份新计划。
输出（--out，默认 outputs/long_run/）：best_grid.npz（网格计划）、best_plan.csv（按秒精修后）、
best.json（指标）、log.jsonl（每轮一行）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import fusion_merge as fm  # noqa: E402
from obsplan import lagrange  # noqa: E402
from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.evolved import load_rule  # noqa: E402
from obsplan.lns import state_from_schedule  # noqa: E402
from obsplan.local_search import improve  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish  # noqa: E402
from obsplan.schedule import Schedule, evaluate, export_csv  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# 新计划的设置轮换：迭代次数、恢复时机偏移、快速局部搜索、是否用伐谋规则
VARIANTS = [
    {"iters": 300, "offset": 0, "fast": False, "rule": True},
    {"iters": 400, "offset": 17, "fast": False, "rule": True},
    {"iters": 300, "offset": 0, "fast": False, "rule": False},
    {"iters": 200, "offset": 29, "fast": False, "rule": True},
    {"iters": 500, "offset": 41, "fast": False, "rule": True},
    {"iters": 150, "offset": 0, "fast": True, "rule": True},
    {"iters": 350, "offset": 23, "fast": False, "rule": False},
    {"iters": 250, "offset": 11, "fast": False, "rule": True},
    {"iters": 600, "offset": 7, "fast": False, "rule": True},
    {"iters": 450, "offset": 33, "fast": False, "rule": False},
]


def ev3(cfg, ds, sch) -> dict:
    e = evaluate(cfg, ds, sch)
    return {"covered": int(e["covered"]), "score": round(float(e["score"]), 3), "combined": round(float(e["combined"]), 6),
            "tasks": int(e["tasks"])}


def grid_schedule(cfg, ds, path) -> Schedule:
    sch, skipped = read_plan(cfg, ds, path)
    assert not skipped, skipped[:3]
    sch.start_s = np.rint(sch.start_s).astype(np.int64)
    sch.dur_s = np.rint(sch.dur_s).astype(np.int64)
    return sch


def save_grid(path: Path, sch: Schedule) -> None:
    np.savez(path, station=sch.station, target=sch.target, start_s=sch.start_s, dur_s=sch.dur_s)


def load_grid(path: Path) -> Schedule:
    z = np.load(path)
    return Schedule(z["station"].astype(np.int64), z["target"].astype(np.int64), z["start_s"].astype(np.int64),
                    z["dur_s"].astype(np.int64), algo="长跑：最好计划")


def make_variant(pb: Problem, k: int, rule):
    """第 k 轮的新计划。每过一圈（len(VARIANTS) 轮）恢复时机再整体错开 13 次迭代，避免逐圈重复同样的计划
    （求解器是确定的，同一设置必然给出同一份计划）。第 0 次迭代总会恢复一次，价格步长才有意义。"""
    v = VARIANTS[k % len(VARIANTS)]
    it = v["iters"]
    offset = (v["offset"] + 13 * (k // len(VARIANTS))) % 50
    recover_at = tuple(sorted({0, *range(offset, it, 50), it - 1}))
    sch, _ = lagrange.solve(pb, iters=it, fast=v["fast"], verbose=False, recover_at=recover_at,
                            rule=rule if v["rule"] else None)
    return sch, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "long_run"))
    ap.add_argument("--anchors", default="G+,E", help="每轮都参与合并的固定参照计划（results/plan_<名字>.csv）")
    ap.add_argument("--start", default="H+", help="没有 --resume 时，初始最好计划取这些里综合最高的一份")
    ap.add_argument("--keep", type=int, default=2, help="合并池里保留最近几份新计划")
    ap.add_argument("--durations", default="4,5,6,7,8")
    ap.add_argument("--shifts", default="-1,1")
    ap.add_argument("--block-limit", type=float, default=30.0)
    ap.add_argument("--block-budget", type=float, default=360.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    rule = load_rule()
    dur = tuple(int(x) for x in args.durations.split(",") if x)
    shifts = tuple(int(x) for x in args.shifts.split(",") if x)

    anchors = [(n, grid_schedule(cfg, ds, ROOT / "results" / f"plan_{n}.csv")) for n in args.anchors.split(",") if n]
    if args.resume and (out / "best_grid.npz").exists():
        best = load_grid(out / "best_grid.npz")
        k0 = sum(1 for _ in open(out / "log.jsonl", encoding="utf-8")) if (out / "log.jsonl").exists() else 0
    else:
        starts = [grid_schedule(cfg, ds, ROOT / "results" / f"plan_{n}.csv") for n in args.start.split(",") if n]
        starts += [s for _, s in anchors]
        best = max(starts, key=lambda s: ev3(cfg, ds, s)["combined"])
        k0 = 0
    assert not structural_violations(pb, best), "初始计划有结构违规"
    best_ev = ev3(cfg, ds, best)
    print("初始最好", best_ev, flush=True)
    recent: list[Schedule] = []
    k = k0
    while time.time() < deadline:
        t0 = time.time()
        var, vcfg = make_variant(pb, k, rule)
        t_var = time.time() - t0
        var_ev = ev3(cfg, ds, var)
        recent = (recent + [var])[-args.keep:]
        sources = [best] + [s for _, s in anchors] + recent
        pool = fm.build_pool(pb, sources, durations=dur, shifts=shifts)
        cur = fm.select_of(pool, fm._cells(pb, best), fm.pool_index(pool))
        t1 = time.time()
        cur, logs = fm.merge_blocks(pb, pool, cur, time_limit=args.block_limit, workers=args.workers, seed=k,
                                    budget_s=args.block_budget, verbose=False)
        merged = fm.to_schedule(pb, pool, cur, "长跑：解合并")
        t2 = time.time()
        st = state_from_schedule(pb, merged)
        improve(pb, st)
        after = st.schedule(pb, "长跑：解合并 + 局部搜索", 0.0)
        t3 = time.time()
        row = {"round": k, "variant": vcfg, "variant_ev": var_ev, "pool": len(pool),
               "merged": ev3(cfg, ds, merged), "after_ls": ev3(cfg, ds, after), "best_before": best_ev["combined"],
               "seconds": {"variant": round(t_var, 1), "merge": round(t2 - t1, 1), "ls": round(t3 - t2, 1)}}
        improved = False
        for cand in (merged, after):
            e = ev3(cfg, ds, cand)
            if e["combined"] > best_ev["combined"] + 1e-9 and not structural_violations(pb, cand):
                best, best_ev, improved = cand, e, True
        row["best_after"] = best_ev["combined"]
        row["improved"] = improved
        if improved:
            save_grid(out / "best_grid.npz", best)
            ref, rinfo = finish(pb, best)
            ref.algo = "长跑：解合并 + 局部搜索 + 按秒精修"
            export_csv(cfg, ds, ref, out / "best_plan.csv")
            ref_ev = ev3(cfg, ds, ref)
            row["refined"] = ref_ev
            (out / "best.json").write_text(json.dumps({"grid": best_ev, "refined": ref_ev, "round": k,
                                                       "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
        with open(out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        print(f"第{k}轮 新计划 {var_ev['combined']:.5f}（{vcfg['iters']}次, {'伐谋' if vcfg['rule'] else '人工'}）"
              f" 合并 {row['merged']['combined']:.5f} 局部搜索 {row['after_ls']['combined']:.5f}"
              f" 最好 {best_ev['combined']:.5f}{' ↑' if improved else ''}"
              f"{'，精修后 ' + format(row['refined']['combined'], '.5f') if improved else ''}"
              f"（{time.time() - t0:.0f} 秒）", flush=True)
        k += 1
    print("结束，最好", best_ev, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
