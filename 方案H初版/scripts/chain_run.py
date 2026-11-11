"""按秒跨站弹射链（obsplan/chain.py）：在一份按秒计划上成批加入高优先级重复观测，变好才接受。

    uv run python scripts/chain_run.py --plan outputs/long_run/best_plan.csv            # 默认最多 3 层弹射
    uv run python scripts/chain_run.py --plan ... --depth 3 --passes 4 --minutes 30

输出（--out，默认 outputs/chain/）：best_plan.csv（按秒计划）、best.json（指标与统计）、log.jsonl（每遍一行）。
全部窗口的按秒真实边界第一次要算几分钟，存在 --bounds（默认 outputs/fable/bounds_all.npz）里复用。
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
from obsplan.chain import Timeline, run_pass  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import refine  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.second_bounds import all_bounds  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def ev3(cfg, ds, sch) -> dict:
    e = evaluate(cfg, ds, sch)
    return {"covered": int(e["covered"]), "score": round(float(e["score"]), 3), "combined": round(float(e["combined"]), 6),
            "tasks": int(e["tasks"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "chain"))
    ap.add_argument("--bounds", default=str(ROOT / "outputs" / "fable" / "bounds_all.npz"))
    ap.add_argument("--depth", type=int, default=2, help="一条链最多顶掉几个任务")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=0.0, help="总时长上限，0 = 不限")
    ap.add_argument("--workers", type=int, default=4, help="算按秒边界用的进程数")
    ap.add_argument("--min-priority", type=float, default=5.0, help="只给优先级分不低于它的目标加重复观测")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="试验用：每遍最多试多少颗目标")
    args = ap.parse_args()
    t_start = time.time()
    deadline = t_start + args.minutes * 60 if args.minutes > 0 else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    sch, skipped = read_plan(cfg, ds, Path(args.plan))
    assert not skipped, skipped[:3]
    sch.start_s = np.rint(sch.start_s).astype(np.int64)
    sch.dur_s = np.rint(sch.dur_s).astype(np.int64)
    e0 = ev3(cfg, ds, sch)
    print(f"起点 {Path(args.plan).name}：{e0}", flush=True)
    wL, wR = all_bounds(pb, Path(args.bounds), workers=args.workers)
    tl = Timeline(pb, wL, wR, sch)
    print(f"时间线 {tl.n_units} 条，对不上窗口的任务 {tl.unmatched} 个，不可行的时间线 {int((~tl.unit_ok).sum())} 条，"
          f"准备用时 {time.time() - t_start:.0f}s", flush=True)
    rng = np.random.default_rng(args.seed)
    base_pts = tl.score_points()
    for k in range(args.passes):
        cand = np.nonzero((pb.P >= args.min_priority) & (tl.tg_cnt < np.minimum(3, pb.max_obs)) & (tl.tg_cnt > 0))[0]
        cand = cand[rng.permutation(len(cand))]
        cand = cand[np.argsort(-pb.P[cand], kind="stable")]
        if args.limit:
            cand = cand[:args.limit]
        r = run_pass(tl, cand, max_depth=args.depth, deadline=deadline)
        r["pass"] = k
        r["stats"] = dict(tl.stats)
        r["points_total"] = tl.score_points() - base_pts
        print(f"第 {k} 遍：试 {r['tried']} 颗，加入 {r['added']} 次，+{r['gain_points']:.1f} 分，{r['seconds']:.0f}s", flush=True)
        with open(out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        cur, L, R = tl.schedule()
        np.savez(out / "state.npz", station=cur.station, target=cur.target, start_s=cur.start_s, dur_s=cur.dur_s, L=L, R=R)
        if r["added"] == 0 or (deadline is not None and time.time() > deadline):
            break
    cur, L, R = tl.schedule()
    e1 = ev3(cfg, ds, cur)
    v1 = {k: len(v) for k, v in structural_violations(pb, cur, by_second=True).items()}
    print(f"弹射链之后：{e1}，结构违规 {v1}", flush=True)
    ref, info = refine(pb, cur, bounds=(L, R))
    e2 = ev3(cfg, ds, ref)
    v2 = {k: len(v) for k, v in structural_violations(pb, ref, by_second=True).items()}
    print(f"全局按秒精修之后：{e2}，结构违规 {v2}", flush=True)
    best, eb, vb = (ref, e2, v2) if (e2["combined"] >= e1["combined"] and not v2) else (cur, e1, v1)
    best.algo = "长跑 + 按秒跨站弹射链"
    export_csv(cfg, ds, best, out / "best_plan.csv")
    (out / "best.json").write_text(json.dumps({
        "start": e0, "after_chain": e1, "after_refine": e2, "best": eb, "violations": vb, "stats": dict(tl.stats),
        "depth": args.depth, "seconds": round(time.time() - t_start, 1), "plan": str(args.plan),
        "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{e0['combined']:.6f} → {eb['combined']:.6f}（{time.time() - t_start:.0f}s），已写 {out / 'best_plan.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
