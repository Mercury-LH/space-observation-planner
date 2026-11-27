"""时间-质量前沿的实验台：把流水线拆成"起点计划"和"弹射链"两步，分别换做法、分别计时。

    uv run python scripts/frontier_lab.py base  --name fast                   # 现行快速档的起点计划（价格协调 + 局部搜索 + 精修）
    uv run python scripts/frontier_lab.py chain --start fast --minutes 1      # 在某份起点计划上跑弹射链，打印得分随时间的轨迹
    uv run python scripts/frontier_lab.py chain --start fast --minutes 1 --profile   # 同上，并统计搜链 / 线性规划 / 提交各占多少时间

起点计划存在 outputs/frontier/lab/<name>.npz（不进仓库），每次实验的汇总追加到 results/frontier/lab_log.jsonl（进仓库）。
这里的用时不含读数据和窗口（约 5 秒），正式数字以 scripts/frontier_bench.py 的端到端结果为准。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obsplan import lagrange  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.evolved import load_rule  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import Schedule, evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

LAB = ROOT / "outputs" / "frontier" / "lab"
LOG = ROOT / "results" / "frontier" / "lab_log.jsonl"


def problem():
    cfg = Config()
    ds = load(cfg)
    win = get_windows(cfg, ds)
    return cfg, ds, Problem.build(cfg, ds, win)


def ev(cfg, ds, sch) -> dict:
    e = evaluate(cfg, ds, sch)
    return {"covered": int(e["covered"]), "score": round(float(e["score"]), 2), "combined": round(float(e["combined"]), 6),
            "tasks": int(e["tasks"]), "repeat_tasks": int(e["repeat_tasks"])}


def save(name: str, sch: Schedule) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    np.savez(LAB / f"{name}.npz", station=sch.station, target=sch.target, start_s=sch.start_s, dur_s=sch.dur_s)


def load_plan(cfg, ds, name: str) -> Schedule:
    p = Path(name)
    if p.suffix == ".csv":                                 # 也可以直接给一份导出的计划
        from obsplan.__main__ import read_plan
        return read_plan(cfg, ds, p if p.is_absolute() else ROOT / p)[0]
    d = np.load(LAB / f"{name}.npz")
    return Schedule(d["station"], d["target"], d["start_s"], d["dur_s"], algo=name)


def log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


def cmd_base(a) -> int:
    cfg, ds, pb = problem()
    for k, v in (kv.split("=", 1) for kv in a.set):       # 实验开关走环境变量（各做法在代码里读）
        os.environ[k] = v
    fast = a.mode == "fast"
    t0 = time.time()
    kw = {"rule": load_rule(), "portfolio": not a.rule_only}
    ls_kw = {}
    if a.rep_minp is not None:
        ls_kw["repeat_min_priority"] = a.rep_minp
    if a.tier_minp is not None:
        ls_kw["tier_min_priority"] = a.tier_minp
    if ls_kw:
        kw["ls_kw"] = ls_kw
    pbs = pb
    if a.m120:                                             # 价格协调和局部搜索只排最短时长，延长留给按秒精修
        import dataclasses
        pbs = dataclasses.replace(pb, m_max=pb.m_min)
    if a.price_rep_minp is not None:                       # 价格迭代里低于这个优先级的目标最多观测 1 次
        kw["max_obs_left"] = np.where(pb.P >= a.price_rep_minp, 3, 1)
    sch, info = lagrange.solve(pbs, iters=a.iters or (150 if fast else 300), fast=fast, verbose=False, **kw)
    t1 = time.time()
    e_grid = ev(cfg, ds, sch)
    from obsplan.refine import finish
    sch2, rinfo = finish(pb, sch)
    t2 = time.time()
    e_ref = ev(cfg, ds, sch2)
    save(a.name + "_grid", sch)
    save(a.name, sch2)
    ls = info["ls"].get("time", {})
    rec = {"kind": "base", "name": a.name, "mode": a.mode, "set": a.set,
           "opts": {k: getattr(a, k) for k in ("m120", "rep_minp", "tier_minp", "price_rep_minp", "iters", "rule_only")},
           "solve_s": round(t1 - t0, 1),
           "refine_s": round(t2 - t1, 1), "ls_time": ls, "grid": e_grid, "refined": e_ref,
           "picked": (info.get("portfolio") or {}).get("picked")}
    log(rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


def cmd_chain(a) -> int:
    cfg, ds, pb = problem()
    for k, v in (kv.split("=", 1) for kv in a.set):
        os.environ[k] = v
    from obsplan import chain
    sch = load_plan(cfg, ds, a.start)
    prof = {}
    if a.profile:                                          # 包一层计时：搜链（numba）/ 线性规划 / 提交
        def timed(cls, name):
            fn = getattr(cls, name)

            def w(*x, **k):
                t = time.perf_counter()
                try:
                    return fn(*x, **k)
                finally:
                    prof[name] = prof.get(name, 0.0) + time.perf_counter() - t
                    prof[name + "_n"] = prof.get(name + "_n", 0) + 1
            setattr(cls, name, w)
        for n in ("lp", "commit", "_propose", "_apply", "units_score"):
            timed(chain.Timeline, n)
        fn = chain._search

        def ws(*x):
            t = time.perf_counter()
            try:
                return fn(*x)
            finally:
                prof["_search"] = prof.get("_search", 0.0) + time.perf_counter() - t
                prof["_search_n"] = prof.get("_search_n", 0) + 1
        chain._search = ws
    trace = []
    if a.trace:                                            # 每次搜链花了多久、成没成（单线程时才有意义）
        fnp = chain.Timeline._propose

        def wp(self, x, tid, scr, max_depth, *rest, **k):
            t = time.perf_counter()
            r = fnp(self, x, tid, scr, max_depth, *rest, **k)
            trace.append((time.perf_counter() - t, r is not None, max_depth))
            return r
        chain.Timeline._propose = wp
    kw = {}
    for k in ("depth", "passes", "threads", "seed", "cycles", "retries", "shift", "repattern_depth"):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    if a.rounds:
        kw["rounds"] = a.rounds
    if a.no_repattern:
        kw["repattern"] = False
    if a.cover is not None:
        kw["cover"] = bool(a.cover)
    if a.pattern is not None:
        kw["pattern"] = bool(a.pattern)
    if a.staged is not None:
        kw["staged"] = bool(a.staged)
    t0 = time.time()
    c0 = time.process_time()
    out, info = chain.improve(pb, sch, minutes=a.minutes, verbose=a.verbose, **kw)
    wall, cpu = time.time() - t0, time.process_time() - c0
    if a.save:
        save(a.save, out)
    traj = [{k: p.get(k) for k in ("spec", "step", "pass", "elapsed", "points", "added", "repatterned", "covered", "dropped",
                                   "depth", "tried", "redo", "copied", "redone", "seconds", "t_propose", "t_redo") if k in p} for p in info["passes"]]
    rec = {"kind": "chain", "start": a.start, "minutes": a.minutes, "set": a.set, "args": kw, "wall_s": round(wall, 1),
           "avg_cores": round(cpu / wall, 2), "from": info["start"], "to": info["best"], "stats": info["stats"],
           "profile": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in prof.items()},
           "stages": info.get("profile"), "passes": traj}
    if trace:
        t = np.array([x[0] for x in trace])
        okm = np.array([x[1] for x in trace])
        srt = np.sort(t)[::-1]
        rec["trace"] = {"n": len(t), "total_s": round(float(t.sum()), 1), "ok_n": int(okm.sum()),
                        "ok_s": round(float(t[okm].sum()), 1), "fail_n": int((~okm).sum()),
                        "fail_s": round(float(t[~okm].sum()), 1),
                        "top1pct_share": round(float(srt[:max(1, len(t) // 100)].sum() / t.sum()), 3),
                        "top10pct_share": round(float(srt[:max(1, len(t) // 10)].sum() / t.sum()), 3),
                        "median_ms": round(float(np.median(t)) * 1e3, 2), "p90_ms": round(float(np.quantile(t, .9)) * 1e3, 2),
                        "p99_ms": round(float(np.quantile(t, .99)) * 1e3, 2), "max_ms": round(float(t.max()) * 1e3, 1)}
    log(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "passes"}, ensure_ascii=False, indent=1))
    for p in traj:
        print("  ", p)
    return 0


def cmd_reloc(a) -> int:
    """单独试"换位置"这一个动作（obsplan/chain.py 的 relocate_pass）：在一份计划上跑若干遍，看每遍搬成多少、涨多少分。"""
    cfg, ds, pb = problem()
    from obsplan import chain
    from obsplan.refine import refine
    from obsplan.second_bounds import all_bounds
    sch = load_plan(cfg, ds, a.start)
    wL, wR = all_bounds(pb, Path(cfg.cache_dir) / f"bounds_all_{cfg.scenario_key()}.npz")
    piece, offset = (int(v) for v in a.spec.split(":"))
    tl = chain.Timeline(pb, wL, wR, sch, piece=piece, offset=offset)
    e0, t0, passes = ev(cfg, ds, sch), time.time(), []
    for k in range(a.passes):
        r = chain.relocate_pass_parallel(tl, threads=a.threads, max_depth=a.depth, min_absorb=a.min_absorb)
        r.update(points=round(tl.score_points(), 1), elapsed=round(time.time() - t0, 1))
        if a.drop:
            r["drop"] = chain.drop_pass(tl)
            r["points_after_drop"] = round(tl.score_points(), 1)
        passes.append(r)
        print("  ", r, flush=True)
        if r["relocated"] == 0:
            break
    cur, L, R, _ = tl.schedule()
    ref, _ = refine(pb, cur, bounds=(L, R))
    e1, e2 = ev(cfg, ds, cur), ev(cfg, ds, ref)
    out = ref if e2["combined"] >= e1["combined"] else cur
    if a.save:
        save(a.save, out)
    rec = {"kind": "reloc", "start": a.start, "depth": a.depth, "spec": a.spec, "min_absorb": a.min_absorb,
           "wall_s": round(time.time() - t0, 1), "from": e0, "to": max(e1, e2, key=lambda e: e["combined"]),
           "stats": dict(tl.stats), "passes": passes}
    log(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "passes"}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--name", required=True)
    b.add_argument("--mode", choices=["fast", "quality"], default="fast")
    b.add_argument("--iters", type=int)
    b.add_argument("--rule-only", action="store_true")
    b.add_argument("--set", action="append", default=[], help="实验开关 KEY=VALUE（环境变量）")
    b.add_argument("--m120", action="store_true", help="价格协调和局部搜索只排 120 秒")
    b.add_argument("--rep-minp", type=float, help="局部搜索的挤让插入重复观测只对不低于这个优先级的目标做（99 = 不做）")
    b.add_argument("--tier-minp", type=float, help="分档补空时低于这个优先级的目标不补重复观测")
    b.add_argument("--price-rep-minp", type=float, help="价格迭代里低于这个优先级的目标最多观测 1 次")
    c = sub.add_parser("chain")
    c.add_argument("--start", required=True, help="起点计划名（outputs/frontier/lab/<名>.npz）或一份计划 CSV")
    c.add_argument("--minutes", type=float, default=1.0)
    c.add_argument("--depth", type=int)
    c.add_argument("--passes", type=int)
    c.add_argument("--threads", type=int)
    c.add_argument("--seed", type=int)
    c.add_argument("--cycles", type=int)
    c.add_argument("--retries", type=int, help="一条链不划算时，禁用出问题的那一步再搜几次")
    c.add_argument("--shift", type=int, help="顶出一个任务后，新任务可插在它前后几个位置以内")
    c.add_argument("--repattern-depth", type=int, dest="repattern_depth")
    c.add_argument("--staged", type=int, help="1 / 0：弹射链是否先浅后深")
    c.add_argument("--pattern", type=int, help="1 / 0：放第二次观测时是否给第三次留位置")
    c.add_argument("--rounds")
    c.add_argument("--no-repattern", action="store_true")
    c.add_argument("--cover", type=int, help="1 / 0：弹射链开头是否先给没覆盖到的目标找链")
    c.add_argument("--profile", action="store_true")
    c.add_argument("--trace", action="store_true", help="记录每次搜链的耗时分布")
    c.add_argument("--verbose", action="store_true")
    c.add_argument("--save", help="把结果计划存成这个名字")
    c.add_argument("--set", action="append", default=[], help="实验开关 KEY=VALUE（环境变量）")
    r = sub.add_parser("reloc")
    r.add_argument("--start", required=True)
    r.add_argument("--depth", type=int, default=1)
    r.add_argument("--passes", type=int, default=3)
    r.add_argument("--spec", default="0:0", help="时间线切法（每段任务数:第一段任务数）")
    r.add_argument("--min-absorb", type=float, default=0.05, dest="min_absorb")
    r.add_argument("--drop", type=int, default=1, help="每遍之后是否再做一遍删重复换延长")
    r.add_argument("--threads", type=int, help="多线程并发数（默认按核数）")
    r.add_argument("--save")
    a = ap.parse_args()
    return {"base": cmd_base, "chain": cmd_chain, "reloc": cmd_reloc}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
