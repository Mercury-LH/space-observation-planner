from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from obsplan.neural_lns.protocol import (assert_catalog_split, build_splits, calibrated_budget,
                                        freeze, load_frozen_protocol, validate_splits)
from obsplan.neural_lns.evaluation import evaluate_campaign, evaluate_pilot, measurement_errors, paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frozen(tmp_path):
    source = ROOT / "experiments/neural-lns-v1"
    for name in ("protocol.json", "splits.json", "feature_schema.json", "IMPLEMENTATION-v1.json", "IMPLEMENTATION-v1.sha256"):
        shutil.copyfile(source / name, tmp_path / name)
    freeze(tmp_path)
    result = load_frozen_protocol(tmp_path)
    # Statistics implementation tests do not need 10k resamples per fixture.
    result["protocol"]["acceptance"]["bootstrap_replicates"] = 128
    return result


def record(frozen, cid, seed, method, score=103.0):
    score = score if method == "neural" else 100.0
    return {"catalog_id": cid, "seed": seed, "method": method, "status": "completed",
            "metrics": {"C": 10, "S": score, "Q": score / 100}, "initial_C": 10,
            "initial_plan_hash": "initial" if method != "h1_multi_bound" else "h1-initial",
            "candidate_generator_hash": "shared-generator", "environment_hash": "same-machine", "code_hash": "same-code",
            "freeze_hash": frozen["freeze"]["freeze_hash"], "model_hash": "a" * 64,
            "calibration_s": [4.0, 4.0, 4.0], "budget_s": 5.0,
            "solve_s": 4.5, "validation_s": 1.0, "end_to_end_s": 6.0,
            "validation": {"legal": True, "geometry_checked": True, "engine": "skyfield", "step_s": 1, "endpoints": True}}


def complete_campaign(frozen):
    ids = frozen["splits"]["splits"]["test"] + [frozen["protocol"]["acceptance"]["application"]["catalog_id"]]
    rows = [record(frozen, cid, seed, method) for cid in ids
            for seed in frozen["protocol"]["seeds"] for method in frozen["protocol"]["methods"]]
    models = {method: {str(seed): "a" * 64 for seed in frozen["protocol"]["seeds"]}
              for method in frozen["protocol"]["model_methods"]}
    return rows, models


def test_split_builder_is_metadata_only_nested_and_month_balanced(frozen):
    specs = json.loads((ROOT / "datasets/synthetic-v0.2/catalog-specs.json").read_text())
    splits = build_splits(specs)
    assert splits == frozen["splits"]
    for n, values in splits["train_subsets"].items():
        months = [splits["catalogs"][c]["month"] for c in values]
        counts = [months.count(m) for m in range(1, 13)]
        assert max(counts) - min(counts) <= 1
        assert len(values) == int(n)
    pilot = [splits["catalogs"][c] for c in splits["pilot_validation"]]
    assert sum(c["sampling_group"] == "core" for c in pilot) == 6
    assert len({c["stress_type"] for c in pilot if c["sampling_group"] == "stress"}) == 6


def test_catalog_leakage_and_derived_state_membership_rejected(frozen):
    splits = copy.deepcopy(frozen["splits"])
    leaked = splits["splits"]["test"][0]
    splits["train_subsets"]["40"][0] = leaked
    with pytest.raises(ValueError, match="subset"):
        validate_splits(splits)
    with pytest.raises(ValueError, match="not in"):
        assert_catalog_split(frozen["splits"], leaked, "train")


def test_freeze_detects_edits_and_refuses_refreezing(tmp_path):
    for name in ("protocol.json", "splits.json", "feature_schema.json"):
        shutil.copyfile(ROOT / "experiments/neural-lns-v1" / name, tmp_path / name)
    first = freeze(tmp_path)
    assert freeze(tmp_path) == first
    path = tmp_path / "protocol.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        freeze(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_protocol(tmp_path)


def test_budget_calibration_is_median_three_and_rejects_bad_times():
    assert calibrated_budget([1., 2., 99.]) == 5.
    assert calibrated_budget([10., 12., 100.]) == 15.
    for times in ([1., 2.], [1., 2., float("nan")], [1., -1., 2.]):
        with pytest.raises(ValueError):
            calibrated_budget(times)


def test_bootstrap_resamples_catalogs_and_is_reproducible():
    a = paired_bootstrap([0.01, 0.03, -0.02], repeats=1000)
    assert a == paired_bootstrap([0.01, 0.03, -0.02], repeats=1000)
    assert a["n_catalogs"] == 3
    assert a["mean"] == pytest.approx(0.02 / 3)


def test_formal_pass_requires_every_group_and_all_seeds(frozen):
    rows, models = complete_campaign(frozen)
    report = evaluate_campaign(rows, frozen, models)
    assert report["adopt"]
    assert report["core"]["relative_S_mean"] == pytest.approx(.03)
    assert report["stress"]["n_catalogs"] == 30
    assert not report["application"]["independent_generalization_test"]
    application = frozen["protocol"]["acceptance"]["application"]["catalog_id"]
    report = evaluate_campaign([r for r in rows if r["catalog_id"] != application], frozen, models)
    assert report["core"]["pass"]
    assert not report["adopt"]
    assert report["application"]["valid_runs"] == 0


def test_failed_neural_stays_in_denominator_and_vetoes_adoption(frozen):
    rows, models = complete_campaign(frozen)
    cid = frozen["splits"]["test_core"][0]
    failed = next(r for r in rows if (r["catalog_id"], r["seed"], r["method"]) == (cid, 2026, "neural"))
    failed.update(status="timeout", solve_s=5.01)
    report = evaluate_campaign(rows, frozen, models)
    assert not report["adopt"]
    assert report["core"]["n_catalogs"] == 60
    detail = next(d for d in report["core"]["details"] if d["catalog_id"] == cid and d["seed"] == 2026)
    assert detail["relative_S"] == -1
    assert detail["effective_neural_S"] == 0
    assert report["core"]["relative_S_mean"] == pytest.approx((179 * .03 - 1) / 180)


def test_stronger_h1_and_pair_mismatch_cannot_be_ignored(frozen):
    rows, models = complete_campaign(frozen)
    for row in rows:
        if row["method"] == "h1_multi_bound":
            row["metrics"]["S"] = 105
    report = evaluate_campaign(rows, frozen, models)
    assert not report["adopt"]
    assert all(d["comparator"] == "h1_multi_bound" for d in report["core"]["details"])
    rows[0]["budget_s"] = 6
    report = evaluate_campaign(rows, frozen, models)
    assert not report["adopt"]
    assert any("paired_budget_s_mismatch" in issue.get("errors", []) for issue in report["issues"])


def test_validation_endpoint_and_timing_fail_closed(frozen):
    row = record(frozen, "x", 2026, "neural")
    row["validation"]["endpoints"] = False
    row["end_to_end_s"] = .1
    errors = measurement_errors(row, expected_freeze=frozen["freeze"]["freeze_hash"])
    assert "independent_validation_missing_or_failed" in errors
    assert "end_to_end_does_not_include_solve_and_validation" in errors


def test_pilot_gate_uses_actual_fixed_rows_and_original_search(frozen):
    rows = [record(frozen, cid, 2026, method) for cid in frozen["splits"]["pilot_validation"]
            for method in ("hplus_extended", "rule", "neural")]
    report = evaluate_pilot(rows, frozen)
    assert report["pass"]
    assert len(report["details"]) == 12
    assert report["core"]["neural_vs_rule_S"] == pytest.approx(.03)
    for row in rows:
        if row["method"] == "hplus_extended":
            row["metrics"]["S"] = 110
    assert not evaluate_pilot(rows, frozen)["pass"]
    assert not evaluate_pilot(rows[:-1], frozen)["pass"]


def test_runtime_freeze_hashes_assets_and_requires_measured_pilot(frozen, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from obsplan.neural_lns import runtime
    from obsplan.neural_lns.protocol import freeze_runtime, file_sha256, load_runtime_freeze
    monkeypatch.setattr(runtime, "environment", lambda instance_id: {"environment_hash": "same-machine"})
    monkeypatch.setattr(runtime, "source_hash", lambda: "same-code")
    models = {}
    normalizer = {"catalog_ids": frozen["splits"]["train_subsets"]["40"], "source_split": "train"}
    for method in sorted(frozen["protocol"]["model_methods"], key=lambda method: method != "static_only"):
        models[method] = {}
        for seed in (2026, 2027, 2028):
            directory = tmp_path / method / str(seed)
            directory.mkdir(parents=True)
            stage = "static" if method == "static_only" else "dynamic"
            config = {"seed": seed, "steps": 10000, "train_catalog_ids": frozen["splits"]["train_subsets"]["40"]}
            identity = {"test": True}
            if stage == "dynamic":
                sample_path = directory / "samples.jsonl"
                sample_path.write_text('{"experiment_scope":"formal"}\n')
                config["samples_path"] = str(sample_path)
                identity["samples_sha256"] = file_sha256(sample_path)
            if method == "neural":
                config["init_checkpoint"] = models["static_only"][str(seed)]
                identity["initialization_sha256"] = file_sha256(config["init_checkpoint"])
            checkpoint = {"config": config, "stage": stage, "step": 5000 if stage == "static" else 10000,
                          "identity": identity, "schema": frozen["feature_schema"],
                          "normalizer": normalizer, "dynamic_normalizer": normalizer if stage == "dynamic" else None}
            torch.save(checkpoint, directory / "checkpoint.pt")
            sha = file_sha256(directory / "checkpoint.pt")
            torch.save({**checkpoint, "step": 10000}, directory / "completion.pt")
            (directory / "latest.json").write_text(json.dumps({"checkpoint": "completion.pt", "step": 10000,
                "sha256": file_sha256(directory / "completion.pt"), "status": "completed_frozen_steps",
                "best_checkpoint": "checkpoint.pt"}))
            (directory / "normalizer.json").write_text(json.dumps(normalizer))
            (directory / "feature_schema.json").write_text(json.dumps(frozen["feature_schema"]))
            if stage == "dynamic":
                (directory / "dynamic_normalizer.json").write_text(json.dumps(normalizer))
                selection_rows = [record(frozen, cid, seed, m) for cid in frozen["splits"]["pilot_validation"]
                                  for m in (method, "rule")]
                for row in selection_rows:
                    row["split"] = "validation"
                    if row["method"] == method:
                        row["model_hash"] = sha
                selection_records = directory / "planner-validation.json"
                selection_records.write_text(json.dumps(selection_rows))
                (directory / "validation_selection.json").write_text(json.dumps({"checkpoint_sha256": sha,
                    "records_path": str(selection_records), "criterion": "scene_mean_relative_S_vs_rule"}))
            else:
                (directory / "validation.jsonl").write_text('{"step":5000,"selected":true}\n')
            models[method][str(seed)] = str(directory / "checkpoint.pt")
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(models))
    calibration = {"complete": True, "selection": "final", "code_hash": "same-code",
                   "environment_hash": "same-machine", "freeze_hash": frozen["freeze"]["freeze_hash"],
                   "budgets": {cid: {"calibration_s": [4, 4, 4], "budget_s": 5}
                               for cid in frozen["splits"]["splits"]["test"] +
                               [frozen["protocol"]["acceptance"]["application"]["catalog_id"]]}}
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    rows = [record(frozen, cid, 2026, method) for cid in frozen["splits"]["pilot_validation"]
            for method in ("hplus_extended", "rule", "neural")]
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(rows))
    lock = tmp_path / "lock.txt"
    lock.write_text("test-only dependency lock")
    kwargs = dict(directory=tmp_path, models_path=models_path, calibration_path=calibration_path,
                  pilot_records_path=pilot_path, dependency_lock=lock, instance_id="test-instance")
    result = freeze_runtime(**kwargs, output=tmp_path / "runtime-freeze.json")
    assert result["pilot_gate_passed"]
    assert result["identity"]["inference_device"] == "cpu"
    assert load_runtime_freeze(tmp_path / "runtime-freeze.json", tmp_path) == result
    assert result["identity"]["models"]["neural"]["2026"] == file_sha256(models["neural"]["2026"])
    rows[-1]["status"] = "timeout"
    pilot_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="evidence hash"):
        load_runtime_freeze(tmp_path / "runtime-freeze.json", tmp_path)
    with pytest.raises(ValueError, match="pilot gate"):
        freeze_runtime(**kwargs, output=tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()
    path = Path(models["neural"]["2026"])
    model = torch.load(path, weights_only=True)
    model["config"]["steps"] = 200
    torch.save(model, path)
    with pytest.raises(ValueError, match="10000 steps"):
        freeze_runtime(**kwargs, output=tmp_path / "must-not-freeze-short-trial.json")
    model["config"]["steps"] = 10000
    model["config"]["experiment_scope"] = "dev-smoke"
    torch.save(model, path)
    with pytest.raises(ValueError, match="development/smoke"):
        freeze_runtime(**kwargs, output=tmp_path / "must-not-freeze-dev-trial.json")


def test_runtime_freeze_rejects_unsupported_inference_device(tmp_path):
    pytest.importorskip("torch")
    from obsplan.neural_lns.protocol import freeze_runtime
    with pytest.raises(ValueError, match="inference_device must be cpu or cuda"):
        freeze_runtime(directory=tmp_path, models_path=tmp_path / "models.json",
                       calibration_path=tmp_path / "calibration.json",
                       pilot_records_path=tmp_path / "pilot.json", dependency_lock=tmp_path / "lock",
                       instance_id="test", output=tmp_path / "output.json", inference_device="mps")
