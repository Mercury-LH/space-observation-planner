"""上界实验：任何合法计划的综合指标都不会超过上界，所以"上界 − 计划"就是这份计划离最优最多还差多少。

两种上界（obsplan/bounds.py）：网络流上界（一次算完）和对偶上界（给每个站的每个 30 秒格定价，迭代收紧）。
两个口径：
  grid    起止都落在 30 秒格点上的计划（网格计划）；
  second  按秒的计划（按秒精修、按秒补空之后的最终计划）——窗口两端各放宽一格并加入短窗口后的放宽问题。

    uv run python scripts/bound_experiment.py --flow                      # 两个口径的网络流上界
    uv run python scripts/bound_experiment.py --dual grid --iters 4000    # 对偶上界；可中断，再跑从断点继续
    uv run python scripts/bound_experiment.py --dual second --iters 4000
    uv run python scripts/bound_experiment.py --verify                    # 用保存的价格重算对偶值，核对 bounds.json

输出：results/bounds.json；价格证书 results/dual_prices_{grid,second}.npz（单精度；记录的对偶值就是用这份
存下来的价格重算的，任何非负价格都给出合法上界，所以舍入不影响合法性）；断点在 outputs/bounds/（不提交）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.bounds import by_second_problem, dual_upper_bound, dual_value, flow_upper_bound  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "bounds.json"
CKPT = ROOT / "outputs" / "bounds"
PLANS = {"H+ 高质量（网格）": "plan_H+.csv", "H+ 高质量（按秒精修）": "plan_H+_refined.csv",
         "长跑优化（按秒精修）": "plan_longrun.csv"}
LABEL = {"grid": "30 秒格点口径", "second": "按秒口径"}


def up6(x: float) -> float:
    """向上取到 6 位小数（记录下来的数仍是合法上界）。"""
    return float(np.ceil(x * 1e6 - 1e-6) / 1e6)


def cert_path(mode: str) -> Path:
    return ROOT / "results" / f"dual_prices_{mode}.npz"


def problems(cfg, ds, modes):
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    out = {}
    for m in modes:
        out[m] = pb if m == "grid" else by_second_problem(pb)
    return pb, out


def plan_values(cfg, ds) -> dict:
    """results/ 里现有计划的综合指标，以及它适用哪个口径（起止全在格点上的算网格计划）。"""
    from obsplan.__main__ import read_plan
    out = {}
    for name, fn in PLANS.items():
        path = ROOT / "results" / fn
        if not path.exists():
            continue
        sch, skipped = read_plan(cfg, ds, path)
        if skipped:
            continue
        on_grid = bool(((sch.start_s % cfg.cell_s == 0) & (sch.dur_s % cfg.cell_s == 0)).all())
        e = evaluate(cfg, ds, sch)
        out[name] = {"file": f"results/{fn}", "scope": "grid" if on_grid else "second", "covered": int(e["covered"]),
                     "score": round(float(e["score"]), 3), "combined": round(float(e["combined"]), 6)}
    return out


def finalize(res: dict) -> None:
    """每个口径取最紧的上界，算各计划离上界的相对差距。"""
    for m in LABEL:
        r = res.get(m)
        if r:
            r["best_ub"] = min(v for k, v in r.items() if k in ("flow_ub", "dual_ub"))
    for p in res.get("plans", {}).values():
        r = res.get(p["scope"])
        if r and "best_ub" in r:
            p["ub"] = r["best_ub"]
            p["gap_pct"] = round(100 * (r["best_ub"] - p["combined"]) / r["best_ub"], 3)
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--dual", choices=list(LABEL))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--target", type=float, help="步长用的目标值（已知最好计划的综合指标；默认取 results/ 里该口径最好的计划）")
    ap.add_argument("--mu0", type=Path, help="起始价格（npz，键 best_mu 或 mu）；默认从断点继续")
    ap.add_argument("--theta", type=float, help="起始步长系数（默认取断点里的，没有断点时 1.0）")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        import numba
        numba.set_num_threads(args.threads)
    cfg = Config()
    ds = load(cfg)
    res = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    res["scenario"] = cfg.scenario_key()
    res["说明"] = ("combined_ub：任何合法计划的综合指标都不超过它。grid = 起止在 30 秒格点上的计划；"
                 "second = 按秒的计划（窗口两端各放宽一格 + 短窗口）。gap_pct = (上界 − 计划) / 上界。")
    modes = list(LABEL) if (args.flow or args.verify) else [args.dual] if args.dual else []
    pb, pbs = problems(cfg, ds, modes)
    res["plans"] = plan_values(cfg, ds)

    if args.flow:
        for m in modes:
            f = flow_upper_bound(pbs[m])
            res.setdefault(m, {}).update({"口径": LABEL[m], "windows": int(len(pbs[m].w_st)),
                                          "flow_ub": up6(f["combined_ub"]), "flow_seconds": round(f["seconds"], 1)})
            finalize(res)

    if args.dual:
        m = args.dual
        ck = CKPT / f"dual_{m}.npz"
        src = args.mu0 if args.mu0 else ck if ck.exists() else None
        mu0, theta = None, 1.0
        if src is not None:
            z = np.load(src)
            mu0 = np.maximum(z["best_mu"] if "best_mu" in z.files else z["mu"], 0.0).astype(np.float64)
            theta = float(z["theta"]) if "theta" in z.files else 1.0
        if args.theta:
            theta = args.theta
        known = [p["combined"] for p in res["plans"].values() if p["scope"] == m]
        target = args.target if args.target else max(known)
        print(f"[{LABEL[m]}] 窗口 {len(pbs[m].w_st)}，步长目标 {target:.6f}，起始步长系数 {theta:.4f}，"
              f"{'从 ' + str(src) + ' 继续' if src is not None else '从零价格开始'}", flush=True)
        r = dual_upper_bound(pbs[m], target=target, iters=args.iters, mu0=mu0, theta=theta, checkpoint=ck)
        mu32 = r["mu"].astype(np.float32)
        value = dual_value(pbs[m], mu32)                        # 证书：按存下来的（单精度）价格重算
        old = res.get(m, {}).get("dual_ub")
        print(f"[{LABEL[m]}] 对偶上界 {value:.6f}（本次 {r['iters']} 次迭代，{r['seconds']:.0f} 秒）", flush=True)
        if old is None or value < old:
            np.savez_compressed(cert_path(m), mu=mu32, combined_ub=value)
            n_prev = res.get(m, {}).get("dual_iters", 0) if src == ck else 0
            res.setdefault(m, {}).update({"口径": LABEL[m], "windows": int(len(pbs[m].w_st)), "dual_ub": up6(value),
                                          "dual_iters": int(n_prev + r["iters"]), "dual_target": target,
                                          "dual_prices": f"results/{cert_path(m).name}"})
        finalize(res)

    if args.verify:
        ok = True
        for m in modes:
            if not cert_path(m).exists():
                continue
            value = dual_value(pbs[m], np.load(cert_path(m))["mu"])
            rec = res.get(m, {}).get("dual_ub")
            good = rec is not None and value <= rec + 1e-9
            ok &= good
            print(f"[{LABEL[m]}] 用保存的价格重算：{value:.6f}；bounds.json 记录 {rec}；{'一致' if good else '不一致'}")
        finalize(res)
        return 0 if ok else 1
    if not (args.flow or args.dual):
        finalize(res)
    print(json.dumps({k: v for k, v in res.items() if k in LABEL or k == "plans"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
