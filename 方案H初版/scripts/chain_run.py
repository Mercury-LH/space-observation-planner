"""按秒跨站弹射链（obsplan/chain.py）：在一份按秒计划上成批加入高优先级重复观测，变好才接受。

    uv run python scripts/chain_run.py --plan outputs/long_run/best_plan.csv
    uv run python scripts/chain_run.py --plan ... --depth 4 --rounds 0:0,40:0,40:20 --minutes 30

--rounds：依次用几种"时间线切法"各做一轮，格式 每段任务数:第一段任务数，0:0 表示每个站每个夜晚一整条。
整条的好处是腾时间可以借到整夜的延长；切段的好处是一条链能在同一站同一夜的不同段里各动一次。
--short：把网格上不够 120 秒、按秒才够的短窗口也作为候选（用了的话结构检查要带上这些窗口，脚本会自动处理并存盘）。

输出（--out，默认 outputs/chain/）：best_plan.csv（按秒计划）、best.json（指标与统计）、log.jsonl（每遍一行）。
全部窗口的按秒真实边界第一次要算十来秒，存在 --bounds（默认 outputs/fable/bounds_all.npz）里复用。
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
from obsplan.chain import Timeline, drop_pass, repattern_pass, run_pass  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import refine  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.second_bounds import all_bounds, short_windows  # noqa: E402
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
    ap.add_argument("--depth", type=int, default=4, help="一条链最多顶掉几个任务")
    ap.add_argument("--rounds", default="0:0,40:0,40:20,0:0", help="依次使用的时间线切法，见文件开头")
    ap.add_argument("--passes", type=int, default=4, help="每种切法最多做几遍")
    ap.add_argument("--minutes", type=float, default=0.0, help="总时长上限，0 = 不限")
    ap.add_argument("--workers", type=int, default=4, help="算按秒边界用的进程数")
    ap.add_argument("--min-priority", type=float, default=5.0, help="只给优先级分不低于它的目标加重复观测")
    ap.add_argument("--shift", type=int, default=4, help="顶出一个任务后，新任务可插在它前后几个位置以内")
    ap.add_argument("--retries", type=int, default=6, help="一条链不划算时，禁用出问题的那一步再搜几次")
    ap.add_argument("--no-drop", action="store_true", help="不允许删掉低优先级的重复观测来腾位置")
    ap.add_argument("--short", action="store_true", help="把按秒才够 120 秒的短窗口也作为候选")
    ap.add_argument("--no-repattern", action="store_true", help="不做「重排节奏」（挪动已有观测给第三次腾出 48 小时间隔）")
    ap.add_argument("--repattern-depth", type=int, default=3)
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
    extra = short_windows(pb) if args.short else None
    if extra is not None:
        print(f"按秒才够长的短窗口 {len(extra['station'])} 个", flush=True)
    rng = np.random.default_rng(args.seed)
    best, best_ev, best_used = sch, e0, None
    stats_all, n_round = {}, 0

    def save(cur, e, used, final=False):
        cur.algo = "长跑 + 按秒跨站弹射链"
        export_csv(cfg, ds, cur, out / "best_plan.csv")
        if used is not None and len(used["station"]):
            np.savez(out / "extra_windows.npz", **used)
        viol = {k: len(v) for k, v in structural_violations(pb, cur, by_second=True, extra_windows=used).items()} if final else None
        (out / "best.json").write_text(json.dumps({
            "start": e0, "best": e, "violations": viol, "stats": stats_all, "depth": args.depth, "rounds": args.rounds,
            "short_windows_used": int(len(used["station"])) if used is not None else 0,
            "seconds": round(time.time() - t_start, 1), "plan": str(args.plan), "final": final,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
        return viol

    for spec in [r for r in args.rounds.split(",") if r]:
        if deadline is not None and time.time() > deadline:
            break
        piece, offset = (int(v) for v in spec.split(":"))
        t0 = time.time()
        tl = Timeline(pb, wL, wR, best, piece=piece, offset=offset, extra_windows=extra)
        print(f"== 切法 {spec}：时间线 {tl.n_units} 条，对不上窗口的任务 {tl.unmatched} 个，"
              f"不可行的时间线 {int((~tl.unit_ok).sum())} 条，准备 {time.time() - t0:.0f}s", flush=True)
        dp = drop_pass(tl)
        for k in range(args.passes):
            cand = np.nonzero((pb.P >= args.min_priority) & (tl.tg_cnt < np.minimum(3, pb.max_obs)) & (tl.tg_cnt > 0))[0]
            cand = cand[rng.permutation(len(cand))]
            cand = cand[np.argsort(-pb.P[cand], kind="stable")]
            if args.limit:
                cand = cand[:args.limit]
            r = run_pass(tl, cand, max_depth=args.depth, deadline=deadline, shift=args.shift, retries=args.retries,
                         drop=not args.no_drop)
            r.update({"round": n_round, "spec": spec, "pass": k, "stats": dict(tl.stats), "dropped_first": dp})
            print(f"切法 {spec} 第 {k} 遍：试 {r['tried']} 颗，加入 {r['added']} 次，+{r['gain_points']:.1f} 分，"
                  f"{r['seconds']:.0f}s", flush=True)
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
            if not args.no_repattern and not (deadline is not None and time.time() > deadline):
                cand = np.nonzero((pb.P >= args.min_priority) & (tl.tg_cnt == 2) & (pb.max_obs >= 3))[0]
                cand = cand[rng.permutation(len(cand))]
                if args.limit:
                    cand = cand[:args.limit]
                before = tl.score_points()
                rp = repattern_pass(tl, cand, max_depth=args.repattern_depth, deadline=deadline, shift=args.shift,
                                    retries=args.retries, drop=not args.no_drop)
                rp.update({"round": n_round, "spec": spec, "pass": k, "stats": dict(tl.stats),
                           "points_check": tl.score_points() - before})
                print(f"切法 {spec} 第 {k} 遍 重排节奏：试 {rp['tried']} 颗，成功 {rp['repatterned']} 颗，"
                      f"+{rp['gain_points']:.1f} 分（复算 +{rp['points_check']:.1f}），{rp['seconds']:.0f}s", flush=True)
                with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rp, ensure_ascii=False, default=float) + "\n")
                r["added"] += rp["repatterned"]
            if r["added"] == 0 or (deadline is not None and time.time() > deadline):
                break
        for k_, v_ in tl.stats.items():
            stats_all[k_] = stats_all.get(k_, 0) + v_
        cur, L, R, used = tl.schedule()
        e1 = ev3(cfg, ds, cur)
        ref, info = refine(pb, cur, bounds=(L, R))          # 全局按秒精修：同目标 48 小时间隔按真实约束重新定时
        e2 = ev3(cfg, ds, ref)
        if e2["combined"] >= e1["combined"]:
            cur, e1 = ref, e2
        print(f"切法 {spec} 结束：{e1}（累计 {time.time() - t_start:.0f}s）", flush=True)
        if e1["combined"] > best_ev["combined"]:
            best, best_ev, best_used = cur, e1, used
            save(best, best_ev, best_used)
        n_round += 1
    viol = save(best, best_ev, best_used, final=True)
    print(f"完成：{e0['combined']:.6f} → {best_ev['combined']:.6f}，任务 {best_ev['tasks']}，结构违规 {viol}"
          f"（{time.time() - t_start:.0f}s），已写 {out / 'best_plan.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
