"""安全余量的取舍：默认余量与保守余量下的可观测目标数与 H（含按秒精修）结果。

    uv run python scripts/margin_experiment.py     # 输出 results/margins.json
"""
from __future__ import annotations

import json
import sys
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
    base = Config()
    rows = []
    for name, cfg in (("当前（默认）", base),
                      ("保守", base.with_(elev_margin=0.1, sun_margin=0.2, sunlit_margin_km=30.0))):
        ds = load(cfg)
        pb = Problem.build(cfg, ds, get_windows(cfg, ds))
        sch, _ = lagrange.solve(pb, iters=300, verbose=False)
        sch, _ = finish(pb, sch)
        ev = evaluate(cfg, ds, sch)
        rows.append({"name": name, "elev_margin": cfg.elev_margin, "sun_margin": cfg.sun_margin,
                     "sunlit_margin_km": cfg.sunlit_margin_km, "observable": int((pb.max_obs > 0).sum()),
                     "covered": ev["covered"], "score": round(ev["score"], 1), "combined": round(ev["combined"], 4)})
        print(rows[-1], flush=True)
    (root / "results" / "margins.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
