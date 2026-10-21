"""备选规划期：以 TLE 历元（约 8 月 3 日）为中心 ±72 小时，即 2026-07-31 00:00 至 08-06 00:00 UTC。
群内答复提到"以提供的时间节点加减 72 小时"，与答疑 #26 的 10/1–10/7 不一致，提前算好以备确认。"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import lagrange  # noqa: E402
from obsplan.bounds import flow_upper_bound  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.schedule import evaluate, export_csv  # noqa: E402
from obsplan.validator import validate  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = Config(start_utc="2026-07-31T00:00:00")
    ds = load(cfg)
    t = time.time()
    win = get_windows(cfg, ds)
    tw = time.time() - t
    pb = Problem.build(cfg, ds, win)
    out = {"start_utc": cfg.start_utc, "windows": int(len(win.k0)), "observable": int((pb.max_obs > 0).sum()),
           "sgp4_failed": int(win.sgp4_failed.sum()), "window_s": round(tw, 1),
           "dark_hours": round(sum((b - a) for r in win.dark for a, b in r) * cfg.cell_s / 3600, 1),
           "dark_hours_by_station": [round(sum(b - a for a, b in r) * cfg.cell_s / 3600, 1) for r in win.dark]}
    for mode, kw in (("快速", dict(iters=150, fast=True)), ("高质量", dict(iters=300))):
        t = time.time()
        sch, info = lagrange.solve(pb, verbose=False, **kw)
        ev = evaluate(cfg, ds, sch)
        out[mode] = {"covered": ev["covered"], "score": ev["score"], "combined": ev["combined"],
                     "repeat_tasks": ev["repeat_tasks"], "runtime_s": round(time.time() - t, 1)}
        print(mode, out[mode], flush=True)
    out["flow_ub"] = flow_upper_bound(pb, verbose=False)["combined_ub"]
    chk = validate(cfg, ds, sch)
    out["validation"] = {"legal": chk["legal"], "violations": chk["violations"]}
    export_csv(cfg, ds, sch, root / "outputs" / "alt_period_plan.csv", chk)
    (root / "results" / "alt_period.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
