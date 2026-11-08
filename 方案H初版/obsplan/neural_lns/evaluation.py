"""Fail-closed, catalog-paired quality acceptance for the registered experiment.

This consumes measurements; it does not certify that a supplied validation report
was truly run. The campaign runner must retain the underlying reports and plans.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from .protocol import calibrated_budget, load_frozen_protocol


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def paired_bootstrap(per_catalog: list[float], *, repeats: int = 10000,
                     seed: int = 20261108) -> dict:
    if not per_catalog or not all(_finite(v) for v in per_catalog) or repeats < 1:
        raise ValueError("bootstrap requires finite catalog-level differences")
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(per_catalog, k=len(per_catalog))) for _ in range(repeats)]
    return {"mean": statistics.fmean(per_catalog), "lower": _quantile(means, 0.025),
            "upper": _quantile(means, 0.975), "confidence": 0.95,
            "unit": "catalog (all seeds averaged before resampling)", "repeats": repeats,
            "seed": seed, "n_catalogs": len(per_catalog)}


def measurement_errors(row: dict, *, expected_freeze: str, expected_model: str | None = None) -> list[str]:
    errors = []
    if row.get("status") != "completed":
        errors.append("not_completed")
    if row.get("freeze_hash") != expected_freeze:
        errors.append("protocol_hash_mismatch")
    if expected_model is not None and row.get("model_hash") != expected_model:
        errors.append("model_hash_mismatch")
    metrics = row.get("metrics", {})
    if not all(_finite(metrics.get(k)) and metrics[k] >= 0 for k in ("C", "S", "Q")):
        errors.append("invalid_quality_metrics")
    elif int(metrics["C"]) != metrics["C"]:
        errors.append("coverage_not_integer")
    if not _finite(row.get("initial_C")) or row["initial_C"] < 0:
        errors.append("invalid_initial_coverage")
    validation = row.get("validation", {})
    if not (validation.get("legal") is True and validation.get("geometry_checked") is True
            and validation.get("engine") == "skyfield"
            and validation.get("step_s") == 1 and validation.get("endpoints") is True):
        errors.append("independent_validation_missing_or_failed")
    try:
        expected_budget = calibrated_budget(row.get("calibration_s", []))
    except (ValueError, TypeError):
        expected_budget = None
        errors.append("invalid_calibration")
    budget = row.get("budget_s")
    if not _finite(budget) or expected_budget is None or not math.isclose(budget, expected_budget, abs_tol=1e-8):
        errors.append("budget_mismatch")
    for field in ("solve_s", "validation_s", "end_to_end_s"):
        if not _finite(row.get(field)) or row[field] < 0:
            errors.append("invalid_" + field)
    if _finite(budget) and _finite(row.get("solve_s")) and row["solve_s"] > budget:
        errors.append("solve_budget_exceeded")
    if all(_finite(row.get(k)) for k in ("solve_s", "validation_s", "end_to_end_s")):
        if row["end_to_end_s"] + 1e-6 < row["solve_s"] + row["validation_s"]:
            errors.append("end_to_end_does_not_include_solve_and_validation")
    for field in ("environment_hash", "code_hash", "initial_plan_hash", "candidate_generator_hash"):
        if not isinstance(row.get(field), str) or not row[field]:
            errors.append("missing_" + field)
    return errors


def pareto(rows: list[dict]) -> list[str]:
    """Caller supplies only valid, same-instance/seed/environment rows."""
    return sorted(row["method"] for row in rows if not any(
        other["metrics"]["Q"] >= row["metrics"]["Q"] and other["solve_s"] <= row["solve_s"]
        and (other["metrics"]["Q"] > row["metrics"]["Q"] or other["solve_s"] < row["solve_s"])
        for other in rows))


def evaluate_pilot(rows: list[dict], frozen: dict) -> dict:
    """Verify the fixed 12-case pilot from actual rows, never a caller boolean.

    The first pilot uses seed 2026. Optional additional seeds must be complete;
    incomplete attempts remain failures. The gate compares complete solve costs,
    including model overhead, under the same calibrated wall budget.
    """
    ids = frozen["splits"]["pilot_validation"]
    required_methods = ("hplus_extended", "rule", "neural")
    seeds = sorted({r.get("seed") for r in rows if r.get("method") in required_methods})
    if not seeds or 2026 not in seeds or not set(seeds) <= {2026, 2027, 2028}:
        return {"pass": False, "issues": ["pilot must include seed 2026 and only registered seeds"]}
    index, issues = {}, []
    for row in rows:
        if row.get("method") not in required_methods:
            continue
        key = (row.get("catalog_id"), row.get("seed"), row.get("method"))
        if key[0] not in ids:
            issues.append({"key": key, "error": "nonpilot_catalog"})
        if key in index:
            issues.append({"key": key, "error": "duplicate_attempt"})
        index[key] = row
    values = []
    for cid in ids:
        for seed in seeds:
            pair = {}
            for method in required_methods:
                key = (cid, seed, method)
                row = index.get(key)
                errors = measurement_errors(row, expected_freeze=frozen["freeze"]["freeze_hash"]) if row else ["missing"]
                if errors:
                    issues.append({"key": key, "errors": errors})
                else:
                    pair[method] = row
            if len(pair) != 3:
                continue
            for field in ("budget_s", "calibration_s", "environment_hash", "code_hash", "initial_plan_hash", "candidate_generator_hash"):
                if len({json.dumps(r[field], sort_keys=True) for r in pair.values()}) != 1:
                    issues.append({"catalog_id": cid, "seed": seed, "error": "paired_" + field + "_mismatch"})
            neural, rule, original = (pair[m] for m in ("neural", "rule", "hplus_extended"))
            if min(rule["metrics"]["S"], original["metrics"]["S"]) <= 0:
                issues.append({"catalog_id": cid, "error": "undefined_relative_gain"})
                continue
            if neural["metrics"]["C"] < max(neural["initial_C"], original["metrics"]["C"]):
                issues.append({"catalog_id": cid, "seed": seed, "error": "coverage_regression"})
            values.append({"catalog_id": cid, "seed": seed,
                           "group": frozen["splits"]["catalogs"][cid]["sampling_group"],
                           "neural_vs_rule_S": neural["metrics"]["S"] / rule["metrics"]["S"] - 1,
                           "rule_vs_original_S": rule["metrics"]["S"] / original["metrics"]["S"] - 1})
    def mean(field, group=None):
        selected = [v[field] for v in values if group is None or v["group"] == group]
        return statistics.fmean(selected) if selected else None
    if len({r.get("environment_hash") for r in index.values()}) != 1:
        issues.append({"error": "pilot_environment_mismatch"})
    if len({r.get("code_hash") for r in index.values()}) != 1:
        issues.append({"error": "pilot_code_mismatch"})
    gain, search_gain = mean("neural_vs_rule_S"), mean("rule_vs_original_S")
    return {"pass": bool(not issues and gain is not None and gain > 0 and search_gain >= 0),
            "neural_vs_rule_S": gain, "rule_vs_original_S": search_gain,
            "core": {k: mean(k, "core") for k in ("neural_vs_rule_S", "rule_vs_original_S")},
            "stress": {k: mean(k, "stress") for k in ("neural_vs_rule_S", "rule_vs_original_S")},
            "catalogs": ids, "seeds": seeds, "issues": issues, "details": values,
            "overhead_gate": "every complete solve including model overhead must fit the same calibrated budget"}


def evaluate_campaign(rows: list[dict], frozen: dict, models: dict, *,
                      allow_partial_report: bool = False) -> dict:
    """Evaluate all 90 test catalogs and the known application, retaining failures.

    ``models`` is frozen before test and maps model method -> string seed -> SHA256.
    Missing required measurements are materialized as failures, never dropped.
    ``allow_partial_report`` only changes the report label; it cannot allow passing.
    """
    protocol, splits = frozen["protocol"], frozen["splits"]
    acceptance = protocol["acceptance"]
    methods, seeds = protocol["methods"], protocol["seeds"]
    application = acceptance["application"]["catalog_id"]
    expected_catalogs = splits["test_core"] + splits["test_stress"] + [application]
    expected_keys = {(c, seed, method) for c in expected_catalogs for seed in seeds for method in methods}
    index, duplicate_keys, extra_rows = {}, set(), []
    for row in rows:
        key = (row.get("catalog_id"), row.get("seed"), row.get("method"))
        if key not in expected_keys:
            extra_rows.append(row)
        elif key in index:
            duplicate_keys.add(key)
        else:
            index[key] = row
    models_complete = all(isinstance(models.get(m, {}).get(str(s)), str)
                          and len(models[m][str(s)]) == 64
                          for m in protocol["model_methods"] for s in seeds)
    valid = {}
    issues = []
    for key in sorted(expected_keys):
        row = index.get(key)
        if row is None:
            errors = ["missing_measurement"]
        else:
            model_hash = models.get(key[2], {}).get(str(key[1]))
            errors = measurement_errors(row, expected_freeze=frozen["freeze"]["freeze_hash"],
                                        expected_model=model_hash)
            if key in duplicate_keys:
                errors.append("duplicate_attempts_require_explicit_frozen_attempt_selection")
        valid[key] = not errors
        if errors:
            issues.append({"catalog_id": key[0], "seed": key[1], "method": key[2], "errors": errors,
                           "record": row})

    # Budget, preparation and environment cannot vary by contender in a pair.
    for cid in expected_catalogs:
        for seed in seeds:
            present = [index[(cid, seed, m)] for m in methods if (cid, seed, m) in index]
            for field in ("budget_s", "calibration_s", "environment_hash", "code_hash", "candidate_generator_hash"):
                values = {json.dumps(r.get(field), sort_keys=True) for r in present}
                if len(values) > 1:
                    issues.append({"catalog_id": cid, "seed": seed, "errors": ["paired_" + field + "_mismatch"]})
                    for m in methods:
                        valid[(cid, seed, m)] = False
            initial_hashes = {r.get("initial_plan_hash") for r in present if r["method"] != "h1_multi_bound"}
            if len(initial_hashes) > 1:
                issues.append({"catalog_id": cid, "seed": seed, "errors": ["paired_initial_plan_hash_mismatch"]})
                for m in methods:
                    valid[(cid, seed, m)] = False

    environments = {r.get("environment_hash") for r in index.values()}
    same_environment = len(environments) == 1 and None not in environments
    codes = {r.get("code_hash") for r in index.values()}
    same_code = len(codes) == 1 and None not in codes
    details, fronts = [], []
    coverage_ok = {}
    for cid in expected_catalogs:
        per_seed = []
        coverage_ok[cid] = True
        for seed in seeds:
            controls = [index[(cid, seed, m)] for m in protocol["non_neural_methods"]
                        if valid.get((cid, seed, m))]
            neural = index.get((cid, seed, "neural"))
            hplus = index.get((cid, seed, "hplus_extended"))
            comparator = max(controls, key=lambda r: (r["metrics"]["S"], r["method"])) if controls else None
            denominator = comparator["metrics"]["S"] if comparator else None
            neural_valid = valid[(cid, seed, "neural")]
            score = neural["metrics"]["S"] if neural_valid else 0.0
            # A missing/zero baseline makes the relative estimand undefined; do not invent one.
            relative = (score - denominator) / denominator if denominator is not None and denominator > 0 else None
            per_seed.append(relative)
            if not (neural_valid and hplus and valid[(cid, seed, "hplus_extended")]
                    and neural["metrics"]["C"] >= max(neural["initial_C"], hplus["metrics"]["C"])):
                coverage_ok[cid] = False
            details.append({"catalog_id": cid, "seed": seed, "comparator": comparator["method"] if comparator else None,
                            "baseline_S": denominator, "effective_neural_S": score,
                            "neural_valid": neural_valid, "relative_S": relative})
            contenders = [index[(cid, seed, m)] for m in methods if valid.get((cid, seed, m))]
            fronts.append({"catalog_id": cid, "seed": seed, "methods": pareto(contenders)})

    def group_summary(ids: list[str], threshold: float, require_ci: bool) -> dict:
        selected = [d for d in details if d["catalog_id"] in ids]
        relative = []
        for cid in ids:
            values = [d["relative_S"] for d in selected if d["catalog_id"] == cid]
            relative.append(statistics.fmean(values) if all(v is not None for v in values) else None)
        complete = all(valid[(c, s, m)] for c in ids for s in seeds for m in methods)
        defined = all(v is not None for v in relative)
        stats = paired_bootstrap(relative, repeats=acceptance["bootstrap_replicates"],
                                 seed=acceptance["bootstrap_seed"]) if defined else None
        mean = stats["mean"] if stats else None
        passes = (complete and models_complete and same_environment and same_code and all(coverage_ok[c] for c in ids)
                  and mean is not None and mean >= threshold
                  and (not require_ci or stats["lower"] > 0))
        return {"n_catalogs": len(ids), "expected_runs": len(ids) * len(seeds) * len(methods),
                "valid_runs": sum(valid[(c, s, m)] for c in ids for s in seeds for m in methods),
                "complete_and_valid": complete, "coverage_non_decreasing": all(coverage_ok[c] for c in ids),
                "relative_S_mean": mean, "paired_catalog_bootstrap": stats,
                "required_mean": threshold, "pass": bool(passes), "details": selected}

    core = group_summary(splits["test_core"], acceptance["core_min_mean_relative_S"], True)
    stress = group_summary(splits["test_stress"], acceptance["stress_min_mean_relative_S"], False)
    known = group_summary([application], acceptance["application"]["min_mean_relative_S"], False)
    stress["by_type"] = {}
    for kind in sorted({splits["catalogs"][c]["stress_type"] for c in splits["test_stress"]}):
        ids = [c for c in splits["test_stress"] if splits["catalogs"][c]["stress_type"] == kind]
        stress["by_type"][kind] = group_summary(ids, 0.0, False)
    stress["per_type_pass_is_not_a_registered_gate"] = True
    stress["interpretation"] = "5 catalogs per type; not a precise failure-probability estimate"
    known["independent_generalization_test"] = False
    utilities = [{"catalog_id": c, "seed": s, "method": m,
                  "J": (4 / 7) * r["metrics"]["Q"] + (3 / 7) * 120 / (120 + r["solve_s"])}
                 for (c, s, m), r in index.items() if valid[(c, s, m)]]
    return {"schema_version": 1, "freeze_hash": frozen["freeze"]["freeze_hash"],
            "report_kind": "partial_diagnostic" if allow_partial_report else "formal_acceptance",
            "adopt": bool(core["pass"] and stress["pass"] and known["pass"] and not extra_rows),
            "core": core, "stress": stress, "application": known,
            "models_complete": models_complete, "same_environment": same_environment, "same_code": same_code,
            "issues": issues, "unexpected_rows": extra_rows, "measurements": rows,
            "pareto_by_instance_seed": fronts, "internal_J": utilities,
            "failure_policy": "failed neural S=0 with denominator retained; any missing/invalid required row vetoes adoption"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="JSON list or JSONL measurements")
    parser.add_argument("--models", type=Path, required=True, help="runtime-freeze.json or method -> seed -> model SHA256 mapping")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    raw = args.records.read_text()
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    model_document = json.loads(args.models.read_text())
    model_hashes = model_document.get("identity", {}).get("models", model_document)
    report = evaluate_campaign(rows, load_frozen_protocol(args.protocol), model_hashes,
                               allow_partial_report=args.partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "adopt": report["adopt"], "issues": len(report["issues"])}))


if __name__ == "__main__":
    main()
