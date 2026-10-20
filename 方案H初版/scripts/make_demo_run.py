"""预先生成一份全量方案 H 结果（含 Skyfield 校验），供产品界面打开即看。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.app_backend import run_planning, validate_run  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

if __name__ == "__main__":
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    run = run_planning(pb, "H", iters=300, progress=lambda f, t: print(f"{f:4.0%} {t}", flush=True) if f in (0.0,) or "局部" in t else None)
    run = validate_run(pb, run)
    path = run.save(Path(__file__).resolve().parents[1] / "outputs" / "app_runs")
    print("saved", path, run.metrics["covered"], round(run.metrics["score"], 1), run.validation)
