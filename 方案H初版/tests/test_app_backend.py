"""界面后端的小工具：演示范围、按范围归一的综合指标、结果显示名称。"""
import numpy as np

from obsplan import app_backend as ab


def _run(pb, algo="H", subset=None, notes=None, run_id="20261110-023251-H"):
    sch = ab.algo_priority(pb)
    m = ab.evaluate(pb.cfg, pb.ds, sch)
    return ab.Run(run_id, algo, {}, subset or {}, sch.station, sch.target, sch.start_s, sch.dur_s, m, 1.0,
                  notes=notes or {})


def test_subset_mask_matches_subset_problem(pb):
    mask = ab.subset_mask(pb.n_targets, 0.2, seed=0)
    small = ab.subset_problem(pb, 0.2, [0, 3, 7], seed=0)
    assert set(np.unique(small.w_tg)) <= set(np.nonzero(mask)[0])
    assert set(np.unique(small.w_st)) <= {0, 3, 7}
    assert 0.15 * pb.n_targets < mask.sum() < 0.25 * pb.n_targets
    assert ab.subset_mask(pb.n_targets, 1.0).all()


def test_scope_metrics(cfg, ds, pb):
    full = _run(pb)
    sc = ab.scope_metrics(cfg, ds, full)
    assert sc["mask"] is None and sc["targets"] == ds.n_targets and sc["combined"] == full.metrics["combined"]
    subset = {"target_frac": 0.2, "stations": list(range(10)), "seed": 0}
    small = _run(ab.subset_problem(pb, 0.2, subset["stations"], 0), subset=subset)
    sc = ab.scope_metrics(cfg, ds, small)
    assert sc["targets"] == int(sc["mask"].sum()) < ds.n_targets
    assert set(np.unique(small.target)) <= set(np.nonzero(sc["mask"])[0])
    # 按参与目标归一：覆盖率不超过 1，综合值高于按全部目标归一的
    assert small.metrics["covered"] <= sc["targets"]
    assert sc["combined"] > small.metrics["combined"]
    # 全部目标都参与（只减站点）时两种归一相同
    st_only = {"target_frac": 1.0, "stations": [0, 1], "seed": 0}
    few = _run(ab.subset_problem(pb, 1.0, st_only["stations"], 0), subset=st_only)
    assert np.isclose(ab.scope_metrics(cfg, ds, few)["combined"], few.metrics["combined"])


def test_optimality_gap(cfg, pb, tmp_path):
    import json
    run = _run(pb)
    run.cfg = json.loads(cfg.to_json())
    path = tmp_path / "bounds.json"
    assert ab.optimality_gap(cfg, run, path) is None                          # 没有上界文件
    path.write_text(json.dumps({"scenario": cfg.scenario_key(), "grid": {"best_ub": 1.0027},
                                "second": {"best_ub": 1.0058}}), encoding="utf-8")
    g = ab.optimality_gap(cfg, run, path)                                     # 网格计划：格点口径的上界
    assert g["scope"] == "grid" and g["ub"] == 1.0027
    assert np.isclose(g["gap_pct"], 100 * (1.0027 - run.metrics["combined"]) / 1.0027) and g["gap_pct"] > 0
    run.notes = {"refine": {"refined": True}}                                 # 按秒精修过：按秒口径的上界
    assert ab.optimality_gap(cfg, run, path)["ub"] == 1.0058
    run.notes, run.start_s = {}, run.start_s + 1                              # 起止不在格点上也算按秒计划
    assert ab.optimality_gap(cfg, run, path)["scope"] == "second"
    assert ab.optimality_gap(cfg.with_(moon_sep_deg=20.0), run, path)["scope"] == "second"   # 月光规避：上界仍成立
    assert ab.optimality_gap(cfg.with_(w_cov=0.7, w_score=0.3), run, path) is None            # 计分口径不同
    assert ab.optimality_gap(cfg.with_(days=3.0), run, path) is None                          # 场景不同
    run.subset = {"target_frac": 0.2, "stations": [0, 1], "seed": 0}
    assert ab.optimality_gap(cfg, run, path) is None                          # 演示子集


def test_run_name(sub):
    assert ab.run_name(_run(sub, notes={"label": "H 高质量"})) == "H 高质量"
    assert ab.run_name(_run(sub, notes={"mode": "快速", "famou_rule": True})) == "H+ 快速 · 11-10 02:32"
    assert ab.run_name(_run(sub, notes={"mode": "高质量", "famou_rule": False})) == "H 高质量 · 11-10 02:32"
    assert ab.run_name(_run(sub, notes={"mode": "高质量", "weather_aware": True})) == "H 按天气 高质量 · 11-10 02:32"
    assert ab.run_name(_run(sub, algo="E", notes={"mode": "快速"}, run_id="20261110-023251-E")) == "E 快速 · 11-10 02:32"
    assert ab.run_name(_run(sub, algo="B", run_id="x")) == "B · x"
