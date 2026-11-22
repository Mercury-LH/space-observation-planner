"""算法效率实验：价格迭代次数与质量/用时，以及快速、高质量两种模式的端到端用时（含窗口计算与按秒精修）。

    uv run python scripts/speed_experiment.py      # 输出 results/speed.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import lagrange  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = Config()
    out = {"iterations": [], "modes": []}
    for mode, iters, fast in (("快速", 150, True), ("高质量", 300, False)):
        t0 = time.time()
        ds = load(cfg)
        win = get_windows(cfg, ds, refresh=True)
        pb = Problem.build(cfg, ds, win)
        t1 = time.time()
        sch, _ = lagrange.solve(pb, iters=iters, fast=fast, verbose=False)
        t2 = time.time()
        sch, _ = finish(pb, sch)
        t3 = time.time()
        ev = evaluate(cfg, ds, sch)
        out["modes"].append({"mode": mode, "iters": iters, "windows_s": round(t1 - t0, 1), "planning_s": round(t2 - t1, 1),
                             "refine_s": round(t3 - t2, 1), "total_s": round(t3 - t0, 1), "covered": ev["covered"],
                             "score": round(ev["score"], 1), "combined": round(ev["combined"], 4)})
        print(out["modes"][-1], flush=True)
    for it in (30, 60, 100, 150, 200, 300):
        t = time.time()
        sch, _ = lagrange.solve(pb, iters=it, verbose=False)
        ts = time.time() - t
        ev = evaluate(cfg, ds, sch)
        out["iterations"].append({"iters": it, "combined_grid": round(ev["combined"], 4), "planning_s": round(ts, 1)})
        print(out["iterations"][-1], flush=True)
    (root / "results" / "speed.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
