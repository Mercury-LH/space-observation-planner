"""算法对比实验：同一数据、同一规则、同一独立校验器（Skyfield）。

用法：
    uv run python run_compare.py              # 默认权重下跑全部方案并校验
    uv run python run_compare.py --sweep      # 另外扫描覆盖/得分权重，画取舍曲线
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from obsplan import cpsat, ga, lagrange
from obsplan.refine import refine
from obsplan.config import Config
from obsplan.data import load
from obsplan.greedy import MANUAL_WEIGHTS, algo_priority, algo_random, algo_weighted
from obsplan.local_search import improve
from obsplan.problem import Problem
from obsplan.schedule import evaluate, export_csv, upper_bound
from obsplan.validator import validate
from obsplan.visibility import get_windows

OUT = Path(__file__).resolve().parent / "results"


def _validate_job(args):
    cfg, sch = args
    ds = load(cfg)
    t = time.time()
    chk = validate(cfg, ds, sch)
    chk["validate_s"] = time.time() - t
    return chk


def run_all(cfg: Config, pb: Problem, h_iters=300, verbose=True):
    """返回 [(Schedule, extra_info)]，按规划方案第八节的顺序。"""
    out = []

    def log(sch):
        ev = evaluate(cfg, pb.ds, sch)
        if verbose:
            print(f"{sch.algo:24s} 覆盖 {ev['covered']:5d} 得分 {ev['score']:8.1f} 综合 {ev['combined']:.4f} 用时 {sch.runtime_s:6.1f}s")

    sch = algo_random(pb, seed=cfg.seed)
    out.append((sch, {}))
    log(sch)
    sch = algo_priority(pb)
    out.append((sch, {}))
    log(sch)
    sch = algo_weighted(pb, MANUAL_WEIGHTS, "C 人工多因素贪心")
    out.append((sch, {}))
    log(sch)

    w_ai, ga_info = ga.train(pb, verbose=verbose)
    t = time.time()
    sch_d, st = algo_weighted(pb, w_ai, "D AI进化权重贪心", return_state=True)
    sch_d.runtime_s += ga_info["train_s"]
    out.append((sch_d, {"weights": w_ai.tolist(), "ga": ga_info}))
    log(sch_d)
    t = time.time()
    ls = improve(pb, st)
    sch = st.schedule(pb, "E AI贪心+局部搜索", sch_d.runtime_s + time.time() - t)
    out.append((sch, {"ls": ls}))
    log(sch)

    sch_g = cpsat.solve(pb, w_ai, hint=sch_d, verbose=verbose)
    sch_g.runtime_s += ga_info["train_s"]
    out.append((sch_g, {}))
    log(sch_g)
    t = time.time()
    st = cpsat.to_state(pb, sch_g)
    ls = improve(pb, st)
    sch = st.schedule(pb, "G+ CP-SAT+局部搜索", sch_g.runtime_s + time.time() - t)
    out.append((sch, {"ls": ls}))
    log(sch)

    sch_h, info = lagrange.solve(pb, iters=h_iters, recover_at=tuple(range(0, h_iters, 50)) + (h_iters - 1,),
                                 verbose=False)
    info.pop("mu", None)
    out.append((sch_h, info))
    log(sch_h)

    # 按秒精修（通用后处理，对所有带局部搜索的方案都做）：网格窗口两端按 1 秒外推到真实边界，线性规划重新分配时长
    for sch, _ in list(out):
        if sch.algo.split()[0] in ("E", "G+", "H"):
            ref, rinfo = refine(pb, sch)
            out.append((ref, {"refine": rinfo, "grid_model": False}))
            log(ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--h-iters", type=int, default=300)
    args = ap.parse_args()

    cfg = Config()
    ds = load(cfg)
    win = get_windows(cfg, ds)
    pb = Problem.build(cfg, ds, win)
    ub = upper_bound(cfg, ds, win, pb.max_obs)
    OUT.mkdir(exist_ok=True)
    print(f"场景 {cfg.scenario_key()}  窗口 {len(pb.w_k0)}  可观测目标 {(pb.max_obs > 0).sum()}  "
          f"站点可用 {ub['capacity_h']:.0f}h  松弛上界(综合) {ub['combined_ub']:.4f}")

    results = run_all(cfg, pb, h_iters=args.h_iters)
    checks = [None] * len(results)
    if not args.no_validate:
        t = time.time()
        with ProcessPoolExecutor(min(len(results), 8)) as ex:
            checks = list(ex.map(_validate_job, [(cfg, s) for s, _ in results]))
        print(f"Skyfield 校验完成 {time.time() - t:.0f}s")

    rows = []
    for (sch, info), chk in zip(results, checks):
        valid = None if chk is None else ~chk["invalid_mask"]
        ev = evaluate(cfg, ds, sch, valid)
        row = {"algo": sch.algo, "runtime_s": round(sch.runtime_s, 1), **ev,
               "legal": None if chk is None else chk["legal"],
               "violations": None if chk is None else chk["violations"],
               "margins": None if chk is None else [chk["min_elev_margin"], chk["max_elev_margin"], chk["sun_margin"]]}
        for k in ("ub_relaxed", "lb_before_ls", "weights", "ls", "refine", "grid_model"):
            if k in info:
                row[k] = info[k]
        rows.append(row)
        slug = sch.algo.split()[0] + ("_refined" if "按秒精修" in sch.algo else "")
        export_csv(cfg, ds, sch, OUT / f"plan_{slug}.csv", chk)
    best_ub = min(ub["combined_ub"], *(r.get("ub_relaxed", np.inf) for r in rows))
    summary = {"scenario": cfg.scenario_key(), "config": json.loads(cfg.to_json()), "upper_bound": ub,
               "lagrange_ub": min(r.get("ub_relaxed", np.inf) for r in rows), "best_ub": best_ub,
               "observable_targets": int((pb.max_obs > 0).sum()), "windows": int(len(pb.w_k0)), "rows": rows}
    (OUT / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"\n{'方案':24s} {'覆盖':>6s} {'得分':>9s} {'重复':>6s} {'综合':>7s} {'距上界':>7s} {'用时s':>7s}  合法")
    for r in rows:
        # 上界是对 30 秒网格模型算的；按秒精修的计划超出了网格模型，不比较距上界
        gap = "—" if r.get("grid_model") is False else f"{(best_ub - r['combined']) / best_ub:7.2%}"
        print(f"{r['algo']:24s} {r['covered']:6d} {r['score']:9.1f} {r['repeat_tasks']:6d} {r['combined']:7.4f} "
              f"{gap:>7s} {r['runtime_s']:7.1f}  {r['legal']} {r['violations'] or ''}")

    if args.sweep:
        sweep(cfg, ds, win)


def sweep(cfg: Config, ds, win):
    """覆盖/得分权重扫描：E（每个权重重新训练 GA）与 H 各自的取舍曲线。"""
    pts = []
    for wc in (1.0, 0.75, 0.5, 0.25, 0.0):
        c2 = cfg.with_(w_cov=wc, w_score=1.0 - wc)
        pb = Problem.build(c2, ds, win)
        w_ai, gi = ga.train(pb, verbose=False)
        t = time.time()
        _, st = algo_weighted(pb, w_ai, "E", return_state=True)
        improve(pb, st)
        e = st.schedule(pb, "E", gi["train_s"] + time.time() - t)
        h, _ = lagrange.solve(pb, iters=300, verbose=False)
        for sch in (e, h):
            ev = evaluate(c2, ds, sch)
            pts.append({"w_cov": wc, "algo": sch.algo[0], "covered": ev["covered"], "score": ev["score"],
                        "runtime_s": sch.runtime_s})
            print(f"  w_cov={wc:.2f} {sch.algo[0]}  覆盖 {ev['covered']}  得分 {ev['score']:.1f}  用时 {sch.runtime_s:.0f}s")
    (OUT / "sweep.json").write_text(json.dumps(pts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
