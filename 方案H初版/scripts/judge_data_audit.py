"""合成数据（synthetic-v0.2 与我们补的 v0.3）审查：和赛题像不像、数据本身对不对、训出来的网络能不能迁移到赛题。

    uv run python scripts/judge_data_audit.py                  # A/B/C 三部分统计
    uv run python scripts/judge_data_audit.py --transfer M.npz # 加 D：模型在赛题上的迁移检验

A. 场景设计参数（v0.2 的 600 个、v0.3 的 180 个，分开统计）与赛题对比：目标数、站数、轨道构成、低轨高度与倾角、优先级、月份、站点纬度。
B. 已生成并算过窗口的场景，实际几何统计（按 v0.2 / v0.3 分开汇总）：低轨高度分布、每颗目标的窗口数、窗口长度、黑夜时长、可观测比例、拥挤度。
C. 数据本身的检查：输入问题、SGP4 失败、重复编号、TLE 历元与规划期的距离。
G. 特征覆盖（--features N）：每套数据各取 N 个已算窗口的场景，用 H 价格算全部候选的 24 维输入，
   看赛题的候选有多少落在该数据集 1%～99% 分位之外（越少越好，说明网络在赛题上不是"外推"）。
D. 迁移：赛题上用 H 价格算特征、长跑最好计划当标签，看模型和人工 / 伐谋规则的 AUC；各输入特征在赛题上相对训练分布的偏移。
输出 results/judge_data_audit.json（文件里已有、这次没重算的部分保留）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obsplan import scenes  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import Windows, get_windows  # noqa: E402

RESULT = ROOT / "results" / "judge_data_audit.json"
MU = 398600.8
RE = 6378.135


def orbit_stats(ds) -> dict:
    n = np.array([float(l[52:63]) for l in ds.line2])
    inc = np.array([float(l[8:16]) for l in ds.line2])
    a = (MU / (n * 2 * np.pi / 86400) ** 2) ** (1 / 3)
    alt = a - RE
    leo, geo = n > 11.25, n <= 1.5
    return {"leo_frac": float(leo.mean()), "meo_frac": float((~leo & ~geo).mean()), "geo_frac": float(geo.mean()),
            "leo_alt_median": float(np.median(alt[leo])) if leo.any() else None,
            "leo_alt_p10": float(np.percentile(alt[leo], 10)) if leo.any() else None,
            "leo_below_550": float((alt[leo] < 550).mean()) if leo.any() else None,
            "leo_inc_mean": float(inc[leo].mean()) if leo.any() else None}


def geometry_stats(cfg, ds, pb) -> dict:
    nw = np.diff(pb.t_ptr)
    dark_h = np.array([sum(b - a + 1 for a, b in runs) for runs in pb.win.dark]) * cfg.cell_s / 3600
    cap = dark_h.sum() * 3600 / cfg.cell_s                        # 黑夜总格数
    obs = pb.max_obs > 0
    need1 = obs.sum() * (pb.m_min + 1)                            # 每颗可观测目标观测一次最短任务所需格数
    return {"n_targets": int(ds.n_targets), "n_stations": int(ds.n_stations),
            "observable_frac": float(obs.mean()), "windows_per_target": float(nw[obs].mean()),
            "window_cells_median": float(np.median(pb.w_k1 - pb.w_k0)),
            "dark_hours_total": float(dark_h.sum()), "dark_hours_min_station": float(dark_h.min()),
            "dark_hours_per_station": float(dark_h.mean()),
            "load_one_obs": float(need1 / cap),                       # >1 表示每颗观测一次都排不下
            "load_three_obs": float(3 * need1 / cap),
            "abs_lat_max": float(np.abs(ds.st_lat).max()), "abs_lat_mean": float(np.abs(ds.st_lat).mean())}


def data_checks(cfg, ds, pb) -> dict:
    from datetime import datetime, timezone
    ep = []
    for l in ds.line1:
        yy, doy = int(l[18:20]), float(l[20:32])
        y = 2000 + yy if yy < 57 else 1900 + yy
        ep.append(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() + (doy - 1) * 86400)
    start = cfg.start_dt.timestamp()
    d = (np.array(ep) - start) / 86400
    return {"issues": len(ds.issues), "sgp4_failed": int(np.asarray(pb.win.sgp4_failed).sum()),
            "duplicate_satnum": int(len(ds.satnum) - len(np.unique(ds.satnum))),
            "epoch_days_from_start_min": float(d.min()), "epoch_days_from_start_max": float(d.max()),
            "priority_values": sorted(float(x) for x in np.unique(ds.priority)),
            "priority_frac": {str(float(v)): float((ds.priority == v).mean()) for v in np.unique(ds.priority)}}


def contest():
    cfg = Config()
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    return cfg, ds, pb


def cached_scenes() -> list[tuple[str, Path, Path]]:
    """(场景, 输入目录, 缓存目录)：judge 流程和泛化检验里已经算过窗口的场景。"""
    out = []
    for base, inputs in ((scenes.OUT, scenes.OUT / "scenes"), (ROOT / "outputs" / "generalization",
                                                              ROOT / "outputs" / "generalization" / "inputs")):
        for d in sorted(inputs.glob("*")):
            c = base / "cache" / d.name
            if (d / "spec.json").exists() and any(c.glob("windows_*.npz")):
                out.append((d.name, d, c))
    seen, uniq = set(), []
    for x in out:
        if x[0] not in seen:
            seen.add(x[0])
            uniq.append(x)
    return uniq


def load_cached(cid: str, d: Path, c: Path):
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    cfg = Config(start_utc=spec["start_utc"], data_dir=str(d), cache_dir=str(c), seed=int(spec["seed"]))
    ds = load(cfg)
    win = Windows.load(Path(c) / f"windows_{cfg.scenario_key()}.npz", ds.n_stations)
    return cfg, ds, Problem.build(cfg, ds, win), spec


def pct_rank(values, x) -> float:
    v = np.asarray([y for y in values if y is not None], float)
    return float((v < x).mean() * 100 + (v == x).mean() * 50) if len(v) else float("nan")


def part_a(contest_orbit: dict, ds_c, dataset: str) -> dict:
    from obsplan.training_data.v2_sampling import STATION_ANCHORS as anchors
    sp = scenes.specs(dataset)
    rows = []
    for s in sp:
        fam = s["family_probabilities"]
        lat = np.abs(anchors[s["station_anchor_ids"], 1])
        rows.append({"n_targets": s["n_targets"], "n_stations": s["n_stations"], "leo_p": fam[0], "meo_p": fam[1],
                     "geo_p": fam[2], "leo_alt_mean": s["sampling_latents"]["leo_altitude_mean_km"],
                     "leo_inc_mean": s["sampling_latents"]["leo_inclination_mean_deg"],
                     "n_high_lat_stations": int((lat >= 46).sum()),
                     "high_p": s["priority_probabilities"][2], "month": int(s["start_utc"][5:7]),
                     "abs_lat_max": float(lat.max())})
    c = {"n_targets": ds_c.n_targets, "n_stations": ds_c.n_stations, "leo_p": contest_orbit["leo_frac"],
         "meo_p": contest_orbit["meo_frac"], "geo_p": contest_orbit["geo_frac"],
         "leo_alt_mean": None, "leo_inc_mean": contest_orbit["leo_inc_mean"],
         "n_high_lat_stations": int((np.abs(ds_c.st_lat) >= 46).sum()),
         "high_p": float((ds_c.priority == ds_c.priority.max()).mean()), "month": 8,
         "abs_lat_max": float(np.abs(ds_c.st_lat).max())}
    used = np.unique(np.concatenate([s["station_anchor_ids"] for s in sp]))
    out = {"scenes": len(sp), "anchor_abs_lat_range": [float(np.abs(anchors[used, 1]).min()),
                                                       float(np.abs(anchors[used, 1]).max())],
           "contest_station_abs_lat": sorted(round(float(x), 1) for x in np.abs(ds_c.st_lat))}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"synthetic_min": float(v.min()), "synthetic_p10": float(np.percentile(v, 10)),
                  "synthetic_median": float(np.median(v)), "synthetic_p90": float(np.percentile(v, 90)),
                  "synthetic_max": float(v.max()), "contest": c[k],
                  "contest_pct_rank": pct_rank(v, c[k]) if c[k] is not None else None}
    return out


def b_summary(rows: list, contest: dict) -> dict:
    summ = {}
    for grp, key in (("orbit", "leo_alt_median"), ("orbit", "leo_below_550"), ("orbit", "meo_frac"), ("orbit", "geo_frac"),
                     ("geometry", "windows_per_target"), ("geometry", "window_cells_median"),
                     ("geometry", "dark_hours_per_station"), ("geometry", "dark_hours_min_station"),
                     ("geometry", "observable_frac"), ("geometry", "load_one_obs"), ("geometry", "load_three_obs"),
                     ("geometry", "abs_lat_max")):
        v = [r[grp][key] for r in rows if r[grp][key] is not None]
        cv = contest[grp][key]
        summ[key] = {"synthetic_min": float(np.min(v)), "synthetic_median": float(np.median(v)),
                     "synthetic_max": float(np.max(v)), "contest": cv, "contest_pct_rank": pct_rank(v, cv)}
    return {"scenes": len(rows), **summ}


def part_d(model_path: str, pb_c, bench_ids: list) -> dict:
    """迁移检验：赛题上的 AUC（标签 = 长跑最好计划）与特征偏移。"""
    from obsplan import judge, lagrange
    from obsplan.famou_h_eval import feature_rows
    from obsplan.greedy import candidates
    from obsplan.schedule import Schedule
    model = judge.load_model(model_path)
    _, info = lagrange.solve(pb_c, iters=300, local_search=False, verbose=False)
    cw, cm = candidates(pb_c)
    X = feature_rows(pb_c, info["mu"], cw, cm)
    Z = judge.inputs(X)
    z = np.load(ROOT / "outputs" / "long_run" / "best_grid.npz")
    best = Schedule(z["station"], z["target"], z["start_s"], z["dur_s"])
    y = judge.labels(pb_c, cw, cm, best).astype(float)
    w = np.ones(len(y))
    res = {"candidates": int(len(y)), "positives": int(y.sum()),
           "auc": {"网络": judge.evaluate_logits(judge.scores(model, X), y, w),
                   "人工规则": judge.evaluate_logits(Z[:, judge.INPUTS.index("manual_u")].astype(float), y, w),
                   "伐谋规则": judge.evaluate_logits(Z[:, judge.INPUTS.index("famou_u")].astype(float), y, w)}}
    # 训练分布：读模型里的训练场景导出数据
    tr = model["meta"].get("train_scenes", [])
    Zt = np.concatenate([np.load(scenes.OUT / "export" / f"{c}.npz")["Z"] for c in tr
                         if (scenes.OUT / "export" / f"{c}.npz").exists()])
    shift = {}
    for j, name in enumerate(judge.INPUTS):
        lo, hi = np.percentile(Zt[:, j], [1, 99])
        sd = Zt[:, j].std() + 1e-9
        shift[name] = {"contest_mean": float(Z[:, j].mean()), "train_mean": float(Zt[:, j].mean()),
                       "shift_in_sd": float((Z[:, j].mean() - Zt[:, j].mean()) / sd),
                       "contest_outside_train_p1_p99": float(((Z[:, j] < lo) | (Z[:, j] > hi)).mean())}
    res["feature_shift"] = shift
    return res


def _h_inputs(pb) -> np.ndarray:
    from obsplan import judge, lagrange
    from obsplan.famou_h_eval import feature_rows
    from obsplan.greedy import candidates
    _, info = lagrange.solve(pb, iters=300, local_search=False, verbose=False)
    cw, cm = candidates(pb)
    return judge.inputs(feature_rows(pb, info["mu"], cw, cm))


def part_g(pb_c, n: int, split: str | None = None) -> dict:
    from obsplan import judge
    Zc = _h_inputs(pb_c)
    by = {}
    for cid, d, c in cached_scenes():
        if split and json.loads((d / "spec.json").read_text(encoding="utf-8"))["split"] != split:
            continue
        by.setdefault(cid.split("-")[0], []).append((cid, d, c))
    rng = np.random.default_rng(0)
    out = {"split": split or "全部", "scenes": {}, "contest_outside_p1_p99": {}}
    for dname, items in sorted(by.items()):
        pick = [items[i] for i in np.linspace(0, len(items) - 1, min(n, len(items))).round().astype(int)]
        Zs = []
        for cid, d, c in pick:
            _, _, pb, _ = load_cached(cid, d, c)
            Z = _h_inputs(pb)
            Zs.append(Z[rng.random(len(Z)) < min(1.0, 400000 / len(Z))])
            print("特征", cid, len(Z), flush=True)
        Z = np.concatenate(Zs)
        out["scenes"][dname] = [x[0] for x in pick]
        row = {}
        for j, name in enumerate(judge.INPUTS):
            lo, hi = np.percentile(Z[:, j], [1, 99])
            row[name] = round(float(((Zc[:, j] < lo) | (Zc[:, j] > hi)).mean()), 4)
        row["任一特征超出"] = round(float(np.mean(np.any([(Zc[:, j] < np.percentile(Z[:, j], 1)) |
                                                  (Zc[:, j] > np.percentile(Z[:, j], 99))
                                                  for j in range(Z.shape[1])], axis=0))), 4)
        out["contest_outside_p1_p99"][dname] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", help="模型文件；给定时做 D 部分（赛题上的迁移检验）")
    ap.add_argument("--features", type=int, default=0, help="G 部分：每套数据取几个场景比较特征覆盖")
    ap.add_argument("--features-split", default=None, help="G 部分只用这个划分的场景（如 train：网络真正学的数据）")
    args = ap.parse_args()
    cfg_c, ds_c, pb_c = contest()
    oc = orbit_stats(ds_c)
    res = {"contest": {"orbit": oc, "geometry": geometry_stats(cfg_c, ds_c, pb_c), "checks": data_checks(cfg_c, ds_c, pb_c)}}
    old = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    res.update({k: v for k, v in old.items() if k not in res})
    res["A_design"] = {d: part_a(oc, ds_c, d) for d in scenes.DATASETS}
    rows = []
    for cid, d, c in cached_scenes():
        cfg, ds, pb, spec = load_cached(cid, d, c)
        rows.append({"catalog_id": cid, "dataset": cid.split("-")[0], "split": spec["split"],
                     "stress_type": spec["stress_type"],
                     "orbit": orbit_stats(ds), "geometry": geometry_stats(cfg, ds, pb), "checks": data_checks(cfg, ds, pb)})
        print(cid, rows[-1]["geometry"]["load_one_obs"], rows[-1]["orbit"]["leo_alt_median"], flush=True)
    res["B_scenes"] = rows
    res["B_summary"] = {}
    for dname in scenes.DATASETS:
        sub = [r for r in rows if r["dataset"] == dname]
        if sub:
            res["B_summary"][dname] = b_summary(sub, res["contest"])
    res["C_checks"] = {"scenes_with_issues": sum(1 for r in rows if r["checks"]["issues"]),
                       "sgp4_failed_total": sum(r["checks"]["sgp4_failed"] for r in rows),
                       "duplicate_satnum_total": sum(r["checks"]["duplicate_satnum"] for r in rows),
                       "epoch_days_from_start": [min(r["checks"]["epoch_days_from_start_min"] for r in rows),
                                                 max(r["checks"]["epoch_days_from_start_max"] for r in rows)],
                       "priority_value_sets": sorted({tuple(r["checks"]["priority_values"]) for r in rows})}
    if args.transfer:
        res["D_transfer"] = part_d(args.transfer, pb_c, [])
    if args.features:
        key = "G_feature_coverage" + (f"_{args.features_split}" if args.features_split else "")
        res[key] = part_g(pb_c, args.features, args.features_split)
        print(json.dumps(res[key]["contest_outside_p1_p99"], ensure_ascii=False, indent=1))
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("B_summary", "C_checks")}, ensure_ascii=False, indent=1))
    if "D_transfer" in res:
        d = res["D_transfer"]
        print(json.dumps(d["auc"], ensure_ascii=False, indent=1))
        big = {k: v for k, v in d["feature_shift"].items() if abs(v["shift_in_sd"]) > 0.5
               or v["contest_outside_train_p1_p99"] > 0.1}
        print("偏移明显的特征：", json.dumps(big, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
