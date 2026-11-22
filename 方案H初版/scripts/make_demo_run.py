"""预先生成产品界面用的结果（含 Skyfield 校验），打开即看。

    uv run python scripts/make_demo_run.py            # 只生成全量方案 H（高质量）
    uv run python scripts/make_demo_run.py --variants # 另外生成快速模式、考虑天气、月光规避 30°、方案 E 的结果
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan.app_backend import run_planning, validate_run  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "app_runs"


def one(cfg, algo, label, **kw):
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    run = run_planning(pb, algo, **kw)
    run.notes["label"] = label
    run = validate_run(pb, run)
    path = run.save(OUT)
    print(label, path.name, run.metrics["covered"], round(run.metrics["score"], 1), run.validation["legal"], flush=True)


if __name__ == "__main__":
    base = Config()
    if "--variants" not in sys.argv:
        one(base, "H", "H 高质量", iters=300)
    else:
        one(base, "H", "H 快速", iters=150, fast=True)
        one(base, "H", "H 考虑天气", iters=300, weather=True)
        one(base.with_(moon_sep_deg=30.0), "H", "H 月光规避 30°", iters=300)
        one(base, "E", "E 快速", fast=True)
