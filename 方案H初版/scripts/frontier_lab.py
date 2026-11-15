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
    sch, info = lagrange.solve(pb, iters=a.iters or (150 if fast else 300), fast=fast, verbose=False, **kw)
    t1 = time.time()
    e_grid = ev(cfg, ds, sch)
    from obsplan.refine import finish
    sch2, rinfo = finish(pb, sch)
    t2 = time.time()
    e_ref = ev(cfg, ds, sch2)
    save(a.name + "_grid", sch)
    save(a.name, sch2)
    ls = info["ls"].get("time", {})
    rec = {"kind": "base", "name": a.name, "mode": a.mode, "set": a.set, "solve_s": round(t1 - t0, 1),
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
    kw = {}
    for k in ("depth", "passes", "threads", "seed"):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    if a.rounds:
        kw["rounds"] = a.rounds
    if a.no_repattern:
        kw["repattern"] = False
    if a.cover is not None:
        kw["cover"] = bool(a.cover)
    t0 = time.time()
    c0 = time.process_time()
    out, info = chain.improve(pb, sch, minutes=a.minutes, verbose=a.verbose, **kw)
    wall, cpu = time.time() - t0, time.process_time() - c0
    if a.save:
        save(a.save, out)
    traj = [{k: p.get(k) for k in ("spec", "step", "pass", "elapsed", "points", "added", "repatterned", "covered",
                                   "tried", "redo", "t_propose", "t_redo") if k in p} for p in info["passes"]]
    rec = {"kind": "chain", "start": a.start, "minutes": a.minutes, "set": a.set, "args": kw, "wall_s": round(wall, 1),
           "avg_cores": round(cpu / wall, 2), "from": info["start"], "to": info["best"], "stats": info["stats"],
           "profile": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in prof.items()}, "passes": traj}
    log(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "passes"}, ensure_ascii=False, indent=1))
    for p in traj:
        print("  ", p)
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
    c = sub.add_parser("chain")
    c.add_argument("--start", required=True, help="起点计划名（outputs/frontier/lab/<名>.npz）或一份计划 CSV")
    c.add_argument("--minutes", type=float, default=1.0)
    c.add_argument("--depth", type=int)
    c.add_argument("--passes", type=int)
    c.add_argument("--threads", type=int)
    c.add_argument("--seed", type=int)
    c.add_argument("--rounds")
    c.add_argument("--no-repattern", action="store_true")
    c.add_argument("--cover", type=int, help="1 / 0：弹射链开头是否先给没覆盖到的目标找链")
    c.add_argument("--profile", action="store_true")
    c.add_argument("--verbose", action="store_true")
    c.add_argument("--save", help="把结果计划存成这个名字")
    c.add_argument("--set", action="append", default=[], help="实验开关 KEY=VALUE（环境变量）")
    a = ap.parse_args()
    return {"base": cmd_base, "chain": cmd_chain}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
