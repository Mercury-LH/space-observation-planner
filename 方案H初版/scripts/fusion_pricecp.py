"""融合实验：价格引导的 CP-SAT 分块（H 的拉格朗日价格 → 候选筛选 → G 的滚动分块 CP-SAT）。

流程：
  1. lagrange.solve 跑 H，拿到收敛的价格 mu（站点 x 格）和 H 自己的（含局部搜索的）计划，后者给分块 CP-SAT 当提示；
  2. 对每个 k_win in (1, 2)：用 obsplan/fusion_pricecp.price_guided_candidates 按"价值 − 价格"筛候选，
     交给同结构的滚动分块 CP-SAT（obsplan/fusion_pricecp.solve）求解；
  3. 网格计划先用 obsplan/checks.structural_violations 检查合法性（必须是空字典），
     再做 local_search.improve，最后 refine.finish 按秒精修，报告精修后的 covered/score/combined；
  4. 跟 H、G+ 的已有结果（方案H初版/results/plan_H.csv、plan_G+.csv）对比，如实报告有没有超过 G+。

    uv run python scripts/fusion_pricecp.py                     # k_win in (1, 2)，每块时限 8 秒
    uv run python scripts/fusion_pricecp.py --k 1 --time-limit 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import fusion_pricecp, lagrange  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.local_search import improve  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_one(pb: Problem, mu, hint, k_win: int, time_limit: float, workers: int, verbose: bool) -> dict:
    t0 = time.time()
    sch_g = fusion_pricecp.solve(pb, mu, k_win=k_win, k_dur=1, time_limit=time_limit, workers=workers,
                                 hint=hint, verbose=verbose, name=f"PriceCP k={k_win}")
    t1 = time.time()
    viol_grid = structural_violations(pb, sch_g)
    ev_grid = evaluate(pb.cfg, pb.ds, sch_g)
    print(f"  [k={k_win}] 网格计划：覆盖 {ev_grid['covered']} 得分 {ev_grid['score']:.1f} 综合 {ev_grid['combined']:.4f} "
          f"用时 {t1 - t0:.0f}s 结构违规 {len(viol_grid)} 项", flush=True)

    st = fusion_pricecp.to_state(pb, sch_g)
    ls_stats = improve(pb, st)
    t2 = time.time()
    sch_ls = st.schedule(pb, f"PriceCP k={k_win} +局部搜索", t2 - t0)
    viol_ls = structural_violations(pb, sch_ls)
    ev_ls = evaluate(pb.cfg, pb.ds, sch_ls)
    print(f"  [k={k_win}] +局部搜索：覆盖 {ev_ls['covered']} 得分 {ev_ls['score']:.1f} 综合 {ev_ls['combined']:.4f} "
          f"用时 {t2 - t0:.0f}s 结构违规 {len(viol_ls)} 项", flush=True)

    sch_ref, rinfo = finish(pb, sch_ls)
    t3 = time.time()
    ev_ref = evaluate(pb.cfg, pb.ds, sch_ref)
    print(f"  [k={k_win}] +按秒精修：覆盖 {ev_ref['covered']} 得分 {ev_ref['score']:.1f} 综合 {ev_ref['combined']:.4f} "
          f"用时 {t3 - t0:.0f}s", flush=True)

    return {
        "k_win": k_win, "time_limit_per_block": time_limit, "workers": workers,
        "cpsat_seconds": round(t1 - t0, 1),
        "grid": {"covered": ev_grid["covered"], "score": round(ev_grid["score"], 1),
                 "combined": round(ev_grid["combined"], 4), "violations": viol_grid,
                 "structural_ok": len(viol_grid) == 0},
        "local_search": {"covered": ev_ls["covered"], "score": round(ev_ls["score"], 1),
                         "combined": round(ev_ls["combined"], 4), "violations": viol_ls,
                         "structural_ok": len(viol_ls) == 0, "stats": ls_stats, "seconds": round(t2 - t1, 1)},
        "refined": {"covered": ev_ref["covered"], "score": round(ev_ref["score"], 1),
                   "combined": round(ev_ref["combined"], 4), "refine_info": rinfo, "seconds": round(t3 - t2, 1)},
        "total_seconds": round(t3 - t0, 1),
        "schedule_refined": sch_ref,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2], help="k_win 设置，可多个")
    ap.add_argument("--time-limit", type=float, default=6.0, help="CP-SAT 每块时限（秒，实际硬上限约 2 倍，见 fusion_pricecp.solve 里的定时兜底）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--h-iters", type=int, default=300)
    ap.add_argument("--budget-s", type=float, default=900.0, help="总时间预算（秒），超了就跳过剩余的 k 设置")
    ap.add_argument("--quiet-cpsat", action="store_true", help="不打印每块 CP-SAT 日志")
    args = ap.parse_args()

    t_start = time.time()
    cfg = Config()
    ds = load(cfg)
    win = get_windows(cfg, ds)
    pb = Problem.build(cfg, ds, win)
    print(f"场景 {cfg.scenario_key()}  窗口 {len(pb.w_k0)}  可观测目标 {(pb.max_obs > 0).sum()}", flush=True)

    print("\n== 第一步：H 价格协调（拿价格 mu，顺带得到一份计划当 CP-SAT 提示）==", flush=True)
    th0 = time.time()
    sch_h, info_h = lagrange.solve(pb, iters=args.h_iters, verbose=False)
    mu = info_h["mu"]
    th1 = time.time()
    ev_h = evaluate(cfg, ds, sch_h)
    print(f"  H：覆盖 {ev_h['covered']} 得分 {ev_h['score']:.1f} 综合 {ev_h['combined']:.4f} 用时 {th1 - th0:.0f}s", flush=True)

    print("\n== 第二步：价格引导 CP-SAT 分块（按 k_win 分别跑）==", flush=True)
    results = []
    skipped = []
    last_run_s = None
    for k in args.k:
        elapsed = time.time() - t_start
        # 用上一次 k 的实际用时估计下一次需要的时间；超预算就跳过，如实记录原因
        if last_run_s is not None and elapsed + last_run_s > args.budget_s:
            print(f"-- k_win={k}：跳过（已用 {elapsed:.0f}s，预计还需约 {last_run_s:.0f}s，超出预算 {args.budget_s:.0f}s）--",
                  flush=True)
            skipped.append({"k_win": k, "reason": f"时间预算不足：已用 {elapsed:.0f}s，预计再需 {last_run_s:.0f}s，"
                            f"预算 {args.budget_s:.0f}s"})
            continue
        print(f"-- k_win={k} --", flush=True)
        tk0 = time.time()
        r = run_one(pb, mu, sch_h, k, args.time_limit, args.workers, verbose=not args.quiet_cpsat)
        last_run_s = time.time() - tk0
        results.append(r)

    if not results:
        print("所有 k_win 设置都没跑完（时间预算不足），没有可报告的融合结果。", flush=True)
        return 1

    # 和已有结果对比（如实报告）
    baseline = {
        "H（精修后）": {"covered": 9987, "score": 28084, "combined": 0.9664},
        "G+（精修后，当前最好）": {"covered": 9992, "score": 28182, "combined": 0.9683},
        "E（精修后）": {"covered": 9999, "score": 26986, "combined": 0.9488},
    }
    print("\n== 对比（covered/score/combined）==", flush=True)
    for name, v in baseline.items():
        print(f"  {name:22s} {v['covered']:6d} {v['score']:9.1f} {v['combined']:.4f}", flush=True)
    best = max(results, key=lambda r: r["refined"]["combined"])
    for r in results:
        tag = "  <= 本次最好" if r is best else ""
        print(f"  PriceCP k={r['k_win']}（精修后）    {r['refined']['covered']:6d} {r['refined']['score']:9.1f} "
              f"{r['refined']['combined']:.4f}{tag}", flush=True)
    beat_gplus = best["refined"]["combined"] > baseline["G+（精修后，当前最好）"]["combined"]
    print(f"\n是否超过 G+ 精修后（{baseline['G+（精修后，当前最好）']['combined']:.4f}）："
          f"{'是' if beat_gplus else '否'}", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    export_csv(cfg, ds, best["schedule_refined"], out_dir / "plan_fusion_pricecp.csv")
    print(f"已导出精修后计划：{out_dir / 'plan_fusion_pricecp.csv'}（取最好的 k_win={best['k_win']}）", flush=True)

    summary = {
        "scenario": cfg.scenario_key(),
        "config": {"h_iters": args.h_iters, "time_limit_per_block": args.time_limit, "workers": args.workers,
                   "k_win_tried": args.k},
        "h_baseline_this_run": {"covered": ev_h["covered"], "score": round(ev_h["score"], 1),
                                "combined": round(ev_h["combined"], 4), "seconds": round(th1 - th0, 1)},
        "existing_results_for_reference": baseline,
        "runs": [{k: v for k, v in r.items() if k != "schedule_refined"} for r in results],
        "skipped": skipped,
        "best_k_win": best["k_win"],
        "beats_g_plus": beat_gplus,
        "total_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "fusion_pricecp.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                                 encoding="utf-8")
    print(f"结果已写入 {out_dir / 'fusion_pricecp.json'}；总用时 {summary['total_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
