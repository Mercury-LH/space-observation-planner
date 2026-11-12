"""更紧的按秒上界（obsplan/second_dual.py）：真实按秒窗口 + 细网格上的拉格朗日对偶上界。

    uv run python scripts/second_dual_bound.py --cell 10 --iters 3000 --target 0.9942

--target 填已知最好计划的综合值（只影响步长，不影响上界的合法性）。从 results/dual_prices_grid.npz（30 秒网格上
迭代好的价格）热启动；--resume 从上次的检查点接着算。结果写 results/bound_second_fine.json，
价格证书存 outputs/fable/dual_second_c<cell>.npz（Dual(pb).value(mu) 一次子问题计算即可复核）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.second_bounds import all_bounds  # noqa: E402
from obsplan.second_dual import dual_upper_bound, fine_problem, warm_start  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=10)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-from", default="", help="从这个检查点的价格出发（步长系数重新给）")
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="方向的惯性系数，0 = 普通次梯度")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bounds", default=str(ROOT / "outputs" / "fable" / "bounds_all.npz"))
    ap.add_argument("--out", default=str(ROOT / "results" / "bound_second_fine.json"))
    args = ap.parse_args()
    t0 = time.time()
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    wL, wR = all_bounds(pb, Path(args.bounds), workers=args.workers)
    pf = fine_problem(pb, wL, wR, args.cell)
    print(f"细网格 {args.cell} 秒：窗口 {len(pf.w_st)} 个，准备用时 {time.time() - t0:.0f}s", flush=True)
    ckpt = ROOT / "outputs" / "fable" / f"dual_second_c{args.cell}{args.tag}.npz"
    theta = args.theta
    if args.start_from:
        z = np.load(args.start_from)
        mu0 = z["mu"]
        print(f"从 {args.start_from} 的价格出发：已有上界 {float(z['best']):.6f}", flush=True)
    elif args.resume and ckpt.exists():
        z = np.load(ckpt)
        mu0, theta = z["mu"], float(z["theta"])
        print(f"从检查点接着算：已有上界 {float(z['best']):.6f}", flush=True)
    else:
        mu0 = warm_start(pf, np.load(ROOT / "results" / "dual_prices_grid.npz")["mu"], cfg.cell_s)
    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None
    r = dual_upper_bound(pf, args.target, iters=args.iters, mu0=mu0, theta=theta, checkpoint=ckpt, deadline=deadline,
                         beta=args.beta, patience=args.patience, decay=args.decay)
    out = {"口径": f"按秒口径（真实按秒窗口，{args.cell} 秒细网格上的对偶上界）", "cell_s": args.cell,
           "windows": int(len(pf.w_st)), "combined_ub": round(r["combined_ub"], 6), "iters": r["iters"],
           "target": args.target, "seconds": round(time.time() - t0, 1), "prices": str(ckpt.relative_to(ROOT)),
           "history": [(int(a), round(b, 6), round(c, 6), round(d, 5)) for a, b, c, d in r["history"]][-40:],
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"按秒上界（细网格 {args.cell} 秒）：{r['combined_ub']:.6f}，迭代 {r['iters']} 次，{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
