"""泛化检验：在侯治民发布的 synthetic-v0.2 测试场景上，比较人工放置规则（H）、伐谋演化的放置规则（H+），
以及产品默认的稳健模式（两条规则各排一份、取综合高的）。

伐谋规则是直接在赛题这一份数据上演化出来的。这里看它在没见过的场景（不同目标数、站点数、轨道构成、月份）
上是否仍然比人工规则好。测试场景从未参与伐谋演化，也没参与任何参数调整。

    # 1. 重建测试场景的输入：与数据集同一个生成器、同一组种子，不用下载 8.9 GB 数据包。
    #    generator 目录里放 feat/neural-training-dataset-v2 分支的 v2_sampling.py、catalog.py 和 catalog-specs.json
    uv run python scripts/generalization_experiment.py --generate --generator DIR --n-core 6 --n-stress 1
    # 2. 逐个场景跑 H、H+、稳健（可中断；再跑只补还缺的）
    uv run python scripts/generalization_experiment.py --mode quality
    # 3. 只汇总
    uv run python scripts/generalization_experiment.py --summary

输出：results/generalization.json（每个场景一行 + 汇总）；输入与窗口缓存在 outputs/generalization/（不提交）。
综合指标 = 0.5 × 覆盖数 / 目标数 + 0.5 × 得分 / 满分参考，按各场景自己的目标数和满分归一化。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obsplan import lagrange  # noqa: E402
from obsplan.checks import structural_violations  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.evolved import load_rule  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.refine import finish, near_windows  # noqa: E402
from obsplan.schedule import evaluate  # noqa: E402
from obsplan.visibility import Windows, compute_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "generalization"
RESULT = ROOT / "results" / "generalization.json"
WORKERS = 8


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select(specs: list[dict], n_core: int, n_stress: int) -> list[dict]:
    test = sorted((s for s in specs if s["split"] == "test"), key=lambda s: s["catalog_id"])
    core = [s for s in test if s["sampling_group"] == "core"][:n_core]
    stress = []
    for kind in sorted({s["stress_type"] for s in test if s["stress_type"]}):
        stress += [s for s in test if s["stress_type"] == kind][:n_stress]
    return core + stress


def generate(gen_dir: Path, n_core: int, n_stress: int) -> None:
    sampling = _module(gen_dir / "v2_sampling.py", "hou_v2_sampling")
    catalog = _module(gen_dir / "catalog.py", "hou_catalog")
    specs = json.loads((gen_dir / "catalog-specs.json").read_text(encoding="utf-8"))
    commit = (gen_dir / "COMMIT").read_text().strip() if (gen_dir / "COMMIT").exists() else ""
    for spec in select(specs, n_core, n_stress):
        d = OUT / "inputs" / spec["catalog_id"]
        if (d / "spec.json").exists():
            continue
        ds, _ = sampling.generate_catalog(spec)
        catalog.write_native_input(d, ds)
        (d / "spec.json").write_text(json.dumps({**spec, "generator_commit": commit}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print("已生成", spec["catalog_id"], spec["n_targets"], "目标", spec["n_stations"], "站",
              spec["start_utc"][:10], spec["stress_type"] or "主体", flush=True)


def _metrics(cfg, ds, pb, sch, seconds, near=None) -> dict:
    e = evaluate(cfg, ds, sch)
    v = structural_violations(pb, sch, by_second=True, extra_windows=near)   # 按秒补空的任务对照真实可见区间
    return {"covered": int(e["covered"]), "score": round(float(e["score"]), 3), "combined": round(float(e["combined"]), 6),
            "tasks": int(e["tasks"]), "repeat_tasks": int(e["repeat_tasks"]), "seconds": round(seconds, 1),
            "violations": int(sum(len(x) for x in v.values()))}


VARIANTS = ("H", "H+", "稳健")     # 人工规则；伐谋规则；两条都排、取综合高的（产品默认）


def run_scene(d: Path, mode: str, rule, row: dict | None = None) -> dict:
    """跑一个场景里 row 还缺的方案（row 为 None 时三个都跑）。"""
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    cfg = Config(start_utc=spec["start_utc"], data_dir=str(d), cache_dir=str(OUT / "cache" / spec["catalog_id"]),
                 seed=int(spec["seed"]))
    ds = load(cfg)
    if ds.issues:
        raise ValueError(f"{spec['catalog_id']} 输入有问题：{ds.issues}")
    t = time.time()
    path = Path(cfg.cache_dir) / f"windows_{cfg.scenario_key()}.npz"
    if path.exists():
        win = Windows.load(path, ds.n_stations)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        win = compute_windows(cfg, ds, workers=WORKERS, verbose=False, keep_short=True)
        win.save(path)
    t_win = time.time() - t
    pb = Problem.build(cfg, ds, win)
    fast = mode == "fast"
    if row is None:
        row = {"catalog_id": spec["catalog_id"], "group": spec["sampling_group"], "stress_type": spec["stress_type"],
               "n_targets": int(spec["n_targets"]), "n_stations": int(spec["n_stations"]), "start": spec["start_utc"][:10],
               "observable": int((pb.max_obs > 0).sum()), "windows_s": round(t_win, 1), "mode": mode}
    for name in VARIANTS:
        if name in row:
            continue
        t = time.time()
        sch, info = lagrange.solve(pb, iters=150 if fast else 300, fast=fast, verbose=False,
                                   rule=None if name == "H" else rule, portfolio=name == "稳健")
        sch, rinfo = finish(pb, sch)
        near = near_windows(pb) if rinfo.get("insert", {}).get("added") else None
        row[name] = _metrics(cfg, ds, pb, sch, time.time() - t, near)
        row[name]["near_added"] = int(rinfo.get("insert", {}).get("added", 0))
        if "portfolio" in info:
            row[name]["picked"] = info["portfolio"]["picked"]
            row[name]["grid_objective"] = {k: round(float(v), 6) for k, v in info["portfolio"]["objective"].items()}
    return row


def _bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 20261110) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _diff_stats(sub: list[dict], a: str, b: str) -> dict:
    """方案 a 相对方案 b（综合指标之差等）。"""
    sub = [r for r in sub if a in r and b in r]
    d = np.array([r[a]["combined"] - r[b]["combined"] for r in sub])
    rel = np.array([(r[a]["score"] - r[b]["score"]) / max(r[b]["score"], 1e-9) for r in sub])
    dc = np.array([r[a]["covered"] - r[b]["covered"] for r in sub])
    lo, hi = _bootstrap_ci(d)
    return {"scenes": len(sub), "combined_diff_mean": round(float(d.mean()), 6) if len(d) else None,
            "combined_diff_ci95": [round(lo, 6), round(hi, 6)],
            "combined_diff_median": round(float(np.median(d)), 6) if len(d) else None,
            "combined_diff_min": round(float(d.min()), 6) if len(d) else None,
            "score_rel_mean": round(float(rel.mean()), 5) if len(rel) else None,
            "covered_diff_mean": round(float(dc.mean()), 2) if len(dc) else None,
            "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()), "equal": int((abs(d) <= 1e-9).sum())}


def summarize(rows: list[dict]) -> dict:
    groups = {"all": rows, "core": [r for r in rows if r["group"] == "core"],
              "stress": [r for r in rows if r["group"] != "core"]}
    out = {}
    for g, sub in groups.items():
        out[g] = {"H+ 相对 H": _diff_stats(sub, "H+", "H"), "稳健 相对 H": _diff_stats(sub, "稳健", "H"),
                  "稳健 相对 H+": _diff_stats(sub, "稳健", "H+")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--generator", type=Path)
    ap.add_argument("--n-core", type=int, default=6)
    ap.add_argument("--n-stress", type=int, default=1)
    ap.add_argument("--mode", choices=["fast", "quality"], default="quality")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--only", help="只跑这些 catalog_id（逗号分隔）")
    ap.add_argument("--workers", type=int, default=8, help="窗口计算的进程数")
    ap.add_argument("--threads", type=int, default=0, help="numba 线程数（0 = 默认）")
    args = ap.parse_args()
    global WORKERS
    WORKERS = args.workers
    if args.threads:
        import numba
        numba.set_num_threads(args.threads)
    if args.generate:
        generate(args.generator, args.n_core, args.n_stress)
        return 0
    res = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {"rows": []}

    def write():
        res["summary"] = {m: summarize([r for r in res["rows"] if r["mode"] == m])
                          for m in sorted({r["mode"] for r in res["rows"]})}
        RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.summary:
        rule = load_rule()
        dirs = sorted(p for p in (OUT / "inputs").glob("*") if (p / "spec.json").exists())
        if args.only:
            keep = set(args.only.split(","))
            dirs = [p for p in dirs if p.name in keep]
        for d in dirs:
            old = next((r for r in res["rows"] if r["catalog_id"] == d.name and r["mode"] == args.mode), None)
            if old is not None and all(v in old for v in VARIANTS):
                continue
            row = run_scene(d, args.mode, rule, old)
            if old is None:
                res["rows"].append(row)
            write()
            h, hp, pf = row["H"], row["H+"], row["稳健"]
            print(f"{row['catalog_id']} {row['n_targets']}目标 {row['n_stations']}站 {row['stress_type'] or '主体'}："
                  f"H {h['combined']:.4f}（{h['covered']}/{h['score']:.0f}） H+ {hp['combined']:.4f}（{hp['covered']}/{hp['score']:.0f}）"
                  f" 差 {hp['combined'] - h['combined']:+.4f}；稳健 {pf['combined']:.4f}（选{'伐谋' if pf.get('picked') == 'rule' else '人工'}）"
                  f"；违规 {h['violations']}/{hp['violations']}/{pf['violations']}", flush=True)
    write()
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
