"""界面后端的小工具：演示范围、按范围归一的综合指标、结果显示名称。"""
import numpy as np
import pytest

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


@pytest.mark.parametrize("fraction,stations,message", [
    (0.2, [], "至少选择"), (0, [0], "目标比例"), (1.1, [0], "目标比例"),
    (float("nan"), [0], "目标比例"), (float("inf"), [0], "目标比例"),
    (0.2, [-1], "所选站点"), (0.2, [2], "所选站点"), (0.2, [0.5], "所选站点"),
])
def test_invalid_subset_is_rejected_before_building(fraction, stations, message):
    from types import SimpleNamespace

    # 不需要可见窗口就能指出配置问题，避免等待计算后才失败。
    with pytest.raises(ValueError, match=message):
        ab.subset_problem(SimpleNamespace(n_stations=2), fraction, stations)


def _saved_minimal_run(tmp_path):
    run = ab.Run("test-run", "H", {}, {}, np.array([0, 1]), np.array([1, 2]), np.array([0, 600]),
                 np.array([120, 120]), {"tasks": 2, "covered": 2, "score": 3.6, "combined": .5}, 1.0)
    return run.save(tmp_path)


def _replace_saved_array(path, **changes):
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_load_saved_run_and_legacy_optional_fields(tmp_path):
    path = _saved_minimal_run(tmp_path)
    run = ab.Run.load(path)
    assert run.run_id == "test-run" and run.validation is None
    assert run.station.tolist() == [0, 1] and run.mu is None
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files if name not in ("per_task", "mu")}
    np.savez_compressed(path, **arrays)
    assert ab.Run.load(path).per_task is None


@pytest.mark.parametrize("changes,message", [
    ({"target": np.array([1])}, "任务数量不一致"),
    ({"station": np.array([[0, 1]])}, "一维"),
    ({"start_s": np.array([0, np.nan])}, "有限数值"),
    ({"dur_s": np.array([120, np.inf])}, "有限数值"),
    ({"target": np.array([1, -1])}, "非负整数"),
    ({"station": np.array([0, .5])}, "非负整数"),
    ({"mu": np.array([[np.nan]])}, "时间价格"),
    ({"per_task": np.zeros((1, 3))}, "校验明细"),
])
def test_load_rejects_damaged_task_arrays(tmp_path, changes, message):
    path = _saved_minimal_run(tmp_path)
    _replace_saved_array(path, **changes)
    with pytest.raises(ValueError, match=message):
        ab.Run.load(path)


def test_load_rejects_invalid_metadata_and_non_npz(tmp_path):
    path = _saved_minimal_run(tmp_path)
    _replace_saved_array(path, meta=np.frombuffer(b"{}", np.uint8))
    with pytest.raises(ValueError, match="结果说明"):
        ab.Run.load(path)
    path.write_bytes(b"incomplete file")
    with pytest.raises(ValueError, match="无法读取结果"):
        ab.Run.load(path)


@pytest.mark.parametrize("field", ["runtime_s", "score"])
def test_load_rejects_nonfinite_summary(tmp_path, field):
    import json

    path = _saved_minimal_run(tmp_path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode())
    if field == "runtime_s":
        meta[field] = float("inf")
    else:
        meta["metrics"][field] = float("nan")
    _replace_saved_array(path, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    with pytest.raises(ValueError, match="无法读取结果"):
        ab.Run.load(path)


def test_load_preserves_failed_geometry_measurements(tmp_path):
    path = _saved_minimal_run(tmp_path)
    run = ab.Run.load(path)
    run.validation = {"legal": False, "violations": {"轨道传播失败或不可信": [0]}}
    run.per_task = np.full((2, 3), np.nan)
    run.save(tmp_path)
    restored = ab.Run.load(path)
    assert not restored.validation["legal"] and np.isnan(restored.per_task).all()


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_failed_save_preserves_previous_result_and_cleans_temporary(tmp_path, monkeypatch, failure_stage):
    from pathlib import Path

    path = _saved_minimal_run(tmp_path)
    previous = path.read_bytes()
    run = ab.Run.load(path)
    run.validation = {"legal": True, "violations": {}, "margins": [1., 2., 3.]}

    if failure_stage == "write":
        def fail_midway(destination, **kwargs):
            # 真实模拟已写入一部分字节后失败；旧的直接覆盖实现会破坏原文件。
            if hasattr(destination, "write"):
                destination.write(b"partial archive")
                destination.flush()
            else:
                Path(destination).write_bytes(b"partial archive")
            raise OSError("simulated partial write failure")

        monkeypatch.setattr(ab.np, "savez_compressed", fail_midway)
    else:
        def fail_replace(source, destination):
            assert ab.Run.load(Path(source)).validation["legal"]
            raise OSError("simulated replacement failure")

        monkeypatch.setattr(ab.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        run.save(tmp_path)
    assert path.read_bytes() == previous
    assert ab.Run.load(path).validation is None
    assert list(tmp_path.iterdir()) == [path]


def test_successful_save_replaces_existing_result_with_readable_update(tmp_path):
    path = _saved_minimal_run(tmp_path)
    run = ab.Run.load(path)
    run.validation = {"legal": True, "violations": {}, "margins": [1., 2., 3.]}
    run.per_task = np.array([[11., 70., -7.], [12., 71., -8.]], dtype=np.float32)
    assert run.save(tmp_path) == path
    updated = ab.Run.load(path)
    assert updated.validation == run.validation
    np.testing.assert_array_equal(updated.per_task, run.per_task)
    np.testing.assert_array_equal(updated.start_s, run.start_s)
    assert list(tmp_path.iterdir()) == [path]
