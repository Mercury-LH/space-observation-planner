"""解合并实验：H、G+、E 三份网格计划 → 候选池 → CP-SAT（G+ 作提示）→ 局部搜索 → 按秒精修。

    uv run python scripts/fusion_merge.py                       # 默认：分块滚动 3 轮，变体 4–8 格 + 平移 ±1 格
    uv run python scripts/fusion_merge.py --sources G+ --tag ablation_gplus_only   # 对照：只用 G+ 自身建池
    uv run python scripts/fusion_merge.py --mode global --global-limit 300
    uv run python scripts/fusion_merge.py --mode blocks --block-h 24 --step-h 12 --block-limit 40
    uv run python scripts/fusion_merge.py --mode stats          # 只看池规模和三方案重叠

输出 results/fusion_merge.json，精修后计划 results/plan_fusion_merge.csv。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from obsplan import fusion_merge as fm  # noqa: E402
from obsplan.__main__ import read_plan  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.lns import state_from_schedule  # noqa: E402
from obsplan.local_search import improve  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ev3(cfg, ds, sch):
    e = evaluate(cfg, ds, sch)
    return {"covered": e["covered"], "score": round(e["score"], 2), "combined": round(e["combined"], 6),
            "tasks": e["tasks"]}


def grid_schedule(cfg, ds, path):
    sch, skipped = read_plan(cfg, ds, path)
    assert not skipped, skipped[:3]
    sch.start_s = np.rint(sch.start_s).astype(np.int64)
    sch.dur_s = np.rint(sch.dur_s).astype(np.int64)
    return sch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="blocks", choices=["stats", "global", "blocks", "both"])
    ap.add_argument("--durations", default="4,5,6,7,8", help="补的变体时长（格）")
    ap.add_argument("--global-limit", type=float, default=240.0)
    ap.add_argument("--block-h", type=float, default=24.0)
    ap.add_argument("--step-h", type=float, default=12.0)
    ap.add_argument("--block-limit", type=float, default=30.0)
    ap.add_argument("--block-budget", type=float, default=300.0)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--shifts", default="-1,1", help="平移变体（格），如 -1,1")
    ap.add_argument("--rounds", type=int, default=3, help=">1 时：局部搜索结果加入池再合并（迭代解合并）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--sources", default="G+,H+,E", help="参与合并的计划；其中综合指标最高的一份作提示")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    T0 = time.time()
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    names = ["G+", "H", "E", "H+"]          # H+ = H + 伐谋放置规则（run_compare.py --add-evolved 生成）
    names = [n for n in names if (ROOT / "results" / f"plan_{n}.csv").exists()]
    plans = {n: grid_schedule(cfg, ds, ROOT / "results" / f"plan_{n}.csv") for n in names}
    out = {"inputs": {}, "args": vars(args)}
    for n, s in plans.items():
        v = structural_violations(pb, s)
        out["inputs"][n] = {**ev3(cfg, ds, s), "violations": {k: len(x) for k, x in v.items()}}
        print(n, out["inputs"][n], flush=True)
    dur = tuple(int(x) for x in args.durations.split(",") if x)
    shifts = tuple(int(x) for x in args.shifts.split(",") if x)
    use = [n for n in args.sources.split(",") if n]
    order = [n for n in names if n in use]
    sources = [plans[n] for n in order]
    full = (1 << len(sources)) - 1
    hint = max(use, key=lambda n: out["inputs"][n]["combined"])
    best = plans[hint]                       # 参与合并的计划里最好的一份（初始提示）
    best_ev = ev3(cfg, ds, best)
    out["hint"] = hint
    out["rounds"] = []
    t_merge = time.time()
    for rd in range(args.rounds):
        pool = fm.build_pool(pb, sources + ([best] if rd > 0 else []), durations=dur, shifts=shifts)
        src = pool.src
        pinfo = {"size": len(pool), "from_plans": int((src > 0).sum()), "variants": int((src == 0).sum()),
                 "shared_all": int(((src & full) == full).sum()),
                 **{f"only_{n}": int((src == (1 << b)).sum()) for b, n in enumerate(order)},
                 "targets": int(len(np.unique(pool.tg)))}
        print(f"第{rd + 1}轮 池", pinfo, flush=True)
        if args.mode == "stats":
            out["pool"] = pinfo
            print(json.dumps(out, ensure_ascii=False, indent=1))
            sys.exit(0)
        idx = fm.pool_index(pool)
        cur = fm.select_of(pool, fm._cells(pb, best), idx)
        ci = fm.coefficients(pb, pool)[1]
        row = {"pool": pinfo, "hint_combined": fm.combined_of(pb, fm.objective(pb, pool, cur), ci)}
        print(f"  提示在池中 combined {row['hint_combined']:.6f}", flush=True)
        if args.mode in ("global", "both"):
            cur, r = fm.merge_global(pb, pool, cur, time_limit=args.global_limit, workers=args.workers, log=args.log)
            row["global"] = {k: v for k, v in r.items() if k != "sel"}
            row["global"]["combined_after"] = fm.combined_of(pb, fm.objective(pb, pool, cur), ci)
            if r["bound"] is not None:
                row["global"]["combined_bound"] = fm.combined_of(pb, r["bound"], ci)
        if args.mode in ("blocks", "both"):
            cur, logs = fm.merge_blocks(pb, pool, cur, block_h=args.block_h, step_h=args.step_h,
                                        time_limit=args.block_limit, workers=args.workers,
                                        budget_s=args.block_budget, passes=args.passes)
            row["blocks"] = logs
        sch = fm.to_schedule(pb, pool, cur, "解合并(H+G+ +E) CP-SAT")
        v = structural_violations(pb, sch)
        row["merged_grid"] = {**ev3(cfg, ds, sch), "violations": {k: len(x) for k, x in v.items()}}
        print("  合并后网格", row["merged_grid"], flush=True)
        assert not v, v
        t = time.time()
        st = state_from_schedule(pb, sch)
        improve(pb, st)
        sch2 = st.schedule(pb, "解合并(H+G+ +E) CP-SAT+局部搜索", 0.0)
        v2 = structural_violations(pb, sch2)
        row["after_ls"] = {**ev3(cfg, ds, sch2), "violations": {k: len(x) for k, x in v2.items()},
                           "seconds": round(time.time() - t, 1)}
        print("  局部搜索后", row["after_ls"], flush=True)
        assert not v2, v2
        out["rounds"].append(row)
        for cand in (sch, sch2):
            e = ev3(cfg, ds, cand)
            if e["combined"] > best_ev["combined"]:
                best, best_ev = cand, e
    out["merge_ls_seconds"] = round(time.time() - t_merge, 1)
    grid_final = best
    out["grid_final"] = best_ev
    assert not structural_violations(pb, grid_final)

    # 来源统计：最终网格任务里，有多少与某份输入计划的任务完全相同（站点、目标、开始、时长）
    keysets = {n: set(zip(*[x.tolist() for x in fm._cells(pb, plans[n])])) for n in names}
    fin = list(zip(*[x.tolist() for x in fm._cells(pb, grid_final)]))
    out["grid_final_origin"] = {
        "tasks": len(fin),
        **{f"same_as_{n}": sum(k in keysets[n] for k in fin) for n in names},
        "only_in_H": sum(k in keysets["H"] and k not in keysets["G+"] and k not in keysets["E"] for k in fin),
        "only_in_E": sum(k in keysets["E"] and k not in keysets["G+"] and k not in keysets["H"] for k in fin),
        "in_none": sum(not any(k in ks for ks in keysets.values()) for k in fin),
    }
    print("来源", out["grid_final_origin"], flush=True)

    t = time.time()
    ref, info = finish(pb, grid_final)
    out["refined"] = {**ev3(cfg, ds, ref), "seconds": round(time.time() - t, 1)}
    print("按秒精修后", out["refined"], flush=True)
    out["runtime_s"] = round(time.time() - T0, 1)
    out["baseline"] = {"G+_grid": out["inputs"]["G+"], "G+_refined": {"covered": 9992, "score": 28182, "combined": 0.9683}}
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = ROOT / "results" / f"plan_fusion_merge{tag}.csv"
    ref.algo = "解合并(H+G+ +E) CP-SAT+局部搜索+按秒精修"
    export_csv(cfg, ds, ref, csv_path)
    out["plan_csv"] = str(csv_path)
    (ROOT / "results" / f"fusion_merge{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                                               encoding="utf-8")
    print(f"总用时 {out['runtime_s']}s", flush=True)
