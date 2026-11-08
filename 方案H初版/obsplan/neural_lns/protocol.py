"""Metadata-only split construction and tamper-evident experiment contracts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


PROTOCOL_VERSION = "neural-lns-v1"
SEEDS = (2026, 2027, 2028)
SUBSET_SIZES = (40, 100, 200, 420)
FROZEN_FILES = ("protocol.json", "splits.json", "feature_schema.json")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tie(value: str, seed: int = 20261108) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_splits(specs: list[dict]) -> dict:
    """Only frozen input metadata is used; no candidate labels or scores are read."""
    catalogs = {}
    for row in specs:
        cid = row["catalog_id"]
        if cid in catalogs:
            raise ValueError(f"duplicate catalog: {cid}")
        catalogs[cid] = {
            "scene_id": cid + "-v0", "split": row["split"],
            "month": int(row["start_utc"][5:7]), "size_bin": row["size_bin"],
            "sampling_group": row["sampling_group"], "stress_type": row["stress_type"],
            "n_targets": row["n_targets"],
        }
    by_split = {split: sorted(cid for cid, row in catalogs.items() if row["split"] == split)
                for split in ("train", "validation", "test")}
    if {k: len(v) for k, v in by_split.items()} != {"train": 420, "validation": 90, "test": 90}:
        raise ValueError("expected complete synthetic-v0.2 metadata, 420/90/90 catalogs")
    if any(catalogs[cid]["sampling_group"] != "core" for cid in by_split["train"]):
        raise ValueError("stress catalogs may not enter train")

    # Cycle months for exact 3/4, 8/9, and 16/17 quotas at the three prefixes.
    # Within each month, deficit allocation tracks its original size proportions.
    months = sorted(range(1, 13), key=lambda m: _tie(f"month:{m}"))
    pools = defaultdict(list)
    for cid in by_split["train"]:
        row = catalogs[cid]
        pools[(row["month"], row["size_bin"])].append(cid)
    for values in pools.values():
        values.sort(key=_tie)
    counts = {key: len(value) for key, value in pools.items()}
    used = Counter()
    month_used = Counter()
    train_order = []
    while len(train_order) < 420:
        for month in months:
            options = [key for key, values in pools.items() if key[0] == month and values]
            if not options:
                continue
            total = sum(n for key, n in counts.items() if key[0] == month)
            key = max(options, key=lambda k: (
                (month_used[month] + 1) * counts[k] / total - used[k], _tie(str(k))))
            train_order.append(pools[key].pop(0))
            used[key] += 1
            month_used[month] += 1

    # Six core cases with two of each size, favoring unrepresented months.
    pilot = []
    month_counts = Counter()
    size_counts = Counter()
    for size in ("small", "medium", "large") * 2:
        choices = [cid for cid in by_split["validation"] if cid not in pilot
                   and catalogs[cid]["sampling_group"] == "core"
                   and catalogs[cid]["size_bin"] == size]
        cid = min(choices, key=lambda c: (month_counts[catalogs[c]["month"]], _tie(c)))
        pilot.append(cid)
        month_counts[catalogs[cid]["month"]] += 1
        size_counts[catalogs[cid]["size_bin"]] += 1
    stress_types = sorted({catalogs[c]["stress_type"] for c in by_split["validation"]
                           if catalogs[c]["sampling_group"] == "stress"})
    for stress in stress_types:
        choices = [cid for cid in by_split["validation"] if catalogs[cid]["stress_type"] == stress]
        cid = min(choices, key=lambda c: (month_counts[catalogs[c]["month"]],
                                         size_counts[catalogs[c]["size_bin"]], _tie(c)))
        pilot.append(cid)
        month_counts[catalogs[cid]["month"]] += 1
        size_counts[catalogs[cid]["size_bin"]] += 1
    result = {
        "version": PROTOCOL_VERSION, "selection_seed": 20261108,
        "selection_uses": "catalog-specs input metadata only; never labels, metrics or outcomes",
        "selection_method": "hash-tied month round-robin; proportional size deficit within month; nested prefixes",
        "pilot_method": "6 core (2 per size) and 1 per each of 6 stress types; favor unused months, hash ties",
        "specs_canonical_sha256": canonical_sha256(specs), "catalogs": catalogs,
        "splits": by_split, "train_subsets": {str(n): train_order[:n] for n in SUBSET_SIZES},
        "pilot_validation": pilot,
        "test_core": [c for c in by_split["test"] if catalogs[c]["sampling_group"] == "core"],
        "test_stress": [c for c in by_split["test"] if catalogs[c]["sampling_group"] == "stress"],
    }
    validate_splits(result)
    return result


def validate_splits(document: dict) -> None:
    split_sets = {k: set(v) for k, v in document["splits"].items()}
    for name, values in document["splits"].items():
        if len(values) != len(split_sets[name]):
            raise ValueError(f"duplicate {name} catalogs")
        for cid in values:
            if document["catalogs"][cid]["split"] != name:
                raise ValueError(f"catalog split mismatch: {cid}")
    if any(split_sets[a] & split_sets[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("catalog split leakage")
    previous = set()
    for n in SUBSET_SIZES:
        ids = document["train_subsets"][str(n)]
        selected = set(ids)
        if len(ids) != n or len(selected) != n or not previous <= selected <= split_sets["train"]:
            raise ValueError(f"invalid nested train subset: {n}")
        if {document["catalogs"][c]["month"] for c in ids} != set(range(1, 13)):
            raise ValueError("train subset does not cover all months")
        previous = selected
    pilot = document["pilot_validation"]
    if len(pilot) != 12 or len(set(pilot)) != 12 or not set(pilot) <= split_sets["validation"]:
        raise ValueError("pilot must contain 12 distinct validation catalogs")
    if len(document["test_core"]) != 60 or len(document["test_stress"]) != 30:
        raise ValueError("test core/stress count mismatch")
    if set(document["test_core"]) & set(document["test_stress"]) or (
        set(document["test_core"]) | set(document["test_stress"])) != split_sets["test"]:
        raise ValueError("test partition mismatch")


def assert_catalog_split(document: dict, catalog_id: str, split: str) -> None:
    if catalog_id not in document["catalogs"] or document["catalogs"][catalog_id]["split"] != split:
        raise ValueError(f"catalog {catalog_id!r} is not in {split!r}")


def calibrated_budget(timings: list[float]) -> float:
    import math
    if len(timings) != 3 or any(not math.isfinite(t) or t <= 0 for t in timings):
        raise ValueError("three positive finite complete H+ timings are required")
    return max(5.0, 1.25 * statistics.median(timings))


def freeze(directory: Path | str) -> dict:
    """Create once. A changed protocol must have a new directory/version."""
    directory = Path(directory)
    document = {"version": PROTOCOL_VERSION,
                "files": {name: file_sha256(directory / name) for name in FROZEN_FILES}}
    document["freeze_hash"] = canonical_sha256(document)
    destination = directory / "FROZEN.json"
    if destination.exists():
        if json.loads(destination.read_text()) != document:
            raise ValueError("frozen protocol changed; register a new version before experiments")
    else:
        with destination.open("x") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return document


def load_frozen_protocol(directory: Path | str) -> dict:
    directory = Path(directory)
    frozen = json.loads((directory / "FROZEN.json").read_text())
    payload = {k: v for k, v in frozen.items() if k != "freeze_hash"}
    if frozen["freeze_hash"] != canonical_sha256(payload):
        raise ValueError("freeze manifest hash mismatch")
    if set(frozen["files"]) != set(FROZEN_FILES):
        raise ValueError("incomplete frozen protocol")
    for name, digest in frozen["files"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"frozen file hash mismatch: {name}")
    result = {Path(name).stem: json.loads((directory / name).read_text()) for name in FROZEN_FILES}
    validate_splits(result["splits"])
    result["freeze"] = frozen
    return result


def validate_dynamic_selection(selection_path: Path | str | dict, document: dict,
                               method: str, seed: int, checkpoint_hash: str) -> dict:
    """Check a dynamic checkpoint's recorded planner validation before test."""
    from .evaluation import measurement_errors
    selection = selection_path if isinstance(selection_path, dict) else json.loads(Path(selection_path).read_text())
    if (selection.get("checkpoint_sha256") != checkpoint_hash
            or selection.get("criterion") != "scene_mean_relative_S_vs_rule"):
        raise ValueError("dynamic validation selection identity/criterion mismatch")
    raw = Path(selection["records_path"]).read_text()
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    if any(r.get("split") != "validation" for r in rows):
        raise ValueError("dynamic checkpoint selection may only use validation planner rows")
    selected = [r for r in rows if r.get("seed") == seed and r.get("method") in (method, "rule")]
    expected = {(cid, name) for cid in document["splits"]["pilot_validation"] for name in (method, "rule")}
    indexed = {(r["catalog_id"], r["method"]): r for r in selected}
    if set(indexed) != expected or len(indexed) != len(selected):
        raise ValueError("dynamic checkpoint selection requires one measured pair for each frozen pilot catalog")
    values = []
    for cid in document["splits"]["pilot_validation"]:
        candidate, rule = indexed[(cid, method)], indexed[(cid, "rule")]
        for row in (candidate, rule):
            errors = measurement_errors(row, expected_freeze=document["freeze"]["freeze_hash"],
                                        expected_model=checkpoint_hash if row is candidate else None)
            if errors:
                raise ValueError("dynamic checkpoint selection contains invalid planner measurements: " + ",".join(errors))
        for field in ("budget_s", "calibration_s", "environment_hash", "code_hash", "initial_plan_hash", "candidate_generator_hash"):
            if candidate[field] != rule[field]:
                raise ValueError("dynamic validation pair mismatch: " + field)
        if rule["metrics"]["S"] <= 0 or candidate["metrics"]["C"] < candidate["initial_C"]:
            raise ValueError("dynamic validation score undefined or coverage decreased")
        values.append(candidate["metrics"]["S"] / rule["metrics"]["S"] - 1)
    return {"scene_mean_relative_S_vs_rule": statistics.fmean(values),
            "catalogs": document["splits"]["pilot_validation"],
            "records_path": str(Path(selection["records_path"]).resolve()),
            "records_sha256": file_sha256(selection["records_path"])}


def freeze_runtime(*, directory: Path | str, models_path: Path | str,
                   calibration_path: Path | str, pilot_records_path: Path | str,
                   dependency_lock: Path | str, instance_id: str,
                   output: Path | str, inference_device: str = "cpu") -> dict:
    """Freeze actual model assets, environment and measured pilot evidence.

    Called on the evaluation host immediately before test. Models are trusted
    local training outputs and loaded with weights_only=True. It never evaluates
    test candidate labels or accepts a manually asserted pilot pass boolean.
    """
    import torch
    from .evaluation import evaluate_pilot
    from .runtime import environment, source_hash

    if inference_device not in ("cpu", "cuda"):
        raise ValueError("inference_device must be cpu or cuda")
    document = load_frozen_protocol(directory)
    models = json.loads(Path(models_path).read_text())
    expected_methods = set(document["protocol"]["model_methods"])
    if set(models) != expected_methods or any(set(v) != {str(s) for s in SEEDS} for v in models.values()):
        raise ValueError("runtime freeze requires all three seeds for all model methods")
    artifacts, model_hashes = {}, {}
    for method, values in models.items():
        artifacts[method], model_hashes[method] = {}, {}
        for seed, name in values.items():
            path = Path(name).resolve()
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            scope = str(checkpoint["config"].get("experiment_scope", "")).lower()
            if "smoke" in scope or "dev" in scope:
                raise ValueError("development/smoke checkpoints cannot enter formal evaluation")
            if checkpoint["config"].get("steps") != document["protocol"]["static_training"]["steps"]:
                raise ValueError("formal main-curve trial must be configured for exactly 10000 steps")
            latest_path = path.parent / "latest.json"
            latest = json.loads(latest_path.read_text())
            completion_path = path.parent / latest["checkpoint"]
            if (latest.get("status") != "completed_frozen_steps" or latest.get("step") != 10000
                    or file_sha256(completion_path) != latest.get("sha256")):
                raise ValueError("formal checkpoint needs same-trial completion evidence at 10000 steps")
            completion = torch.load(completion_path, map_location="cpu", weights_only=True)
            if (completion.get("step") != 10000 or completion["config"] != checkpoint["config"]
                    or completion["identity"] != checkpoint["identity"] or not 0 < checkpoint.get("step", 0) <= 10000):
                raise ValueError("selected checkpoint and completed 10000-step trial differ")
            if checkpoint["config"]["seed"] != int(seed):
                raise ValueError(f"checkpoint seed mismatch: {method}/{seed}")
            train_catalogs = set(checkpoint["config"].get("train_catalog_ids", []))
            if not any(train_catalogs == set(ids) for ids in document["splits"]["train_subsets"].values()):
                raise ValueError("formal checkpoint must use an exact preregistered train subset")
            required_stage = "static" if method == "static_only" else "dynamic"
            if checkpoint["stage"] != required_stage:
                raise ValueError(f"checkpoint stage mismatch: {method}/{seed}")
            if method == "neural" and not checkpoint["config"].get("init_checkpoint"):
                raise ValueError("neural arm requires stage A initialization")
            if method == "neural":
                initialization_hash = file_sha256(checkpoint["config"]["init_checkpoint"])
                if (initialization_hash != checkpoint["identity"].get("initialization_sha256")
                        or initialization_hash != file_sha256(models["static_only"][seed])):
                    raise ValueError("neural initialization must match the same-seed registered stage-A ablation")
            if method == "dynamic_random_init" and checkpoint["config"].get("init_checkpoint"):
                raise ValueError("random-initialization ablation cannot use stage A initialization")
            schema_path, normalizer_path = path.parent / "feature_schema.json", path.parent / "normalizer.json"
            schema = json.loads(schema_path.read_text())
            normalizer = json.loads(normalizer_path.read_text())
            if schema != document["feature_schema"] or schema != checkpoint["schema"]:
                raise ValueError("checkpoint/exported/frozen feature schema mismatch")
            if normalizer != checkpoint["normalizer"]:
                raise ValueError("exported normalizer differs from checkpoint")
            if normalizer.get("source_split") != "train" or not set(normalizer["catalog_ids"]) <= train_catalogs:
                raise ValueError("normalizer exceeds the declared train subset")
            for cid in normalizer["catalog_ids"]:
                assert_catalog_split(document["splits"], cid, "train")
            files = {"checkpoint": str(path), "schema": str(schema_path), "normalizer": str(normalizer_path)}
            if method == "neural":
                files["initialization_checkpoint"] = str(Path(checkpoint["config"]["init_checkpoint"]).resolve())
            files.update(completion_checkpoint=str(completion_path), latest=str(latest_path))
            if required_stage == "static":
                validation_path = path.parent / "validation.jsonl"
                validations = [json.loads(line) for line in validation_path.read_text().splitlines() if line.strip()]
                if latest.get("best_checkpoint") != path.name or not any(
                        v.get("step") == checkpoint["step"] and v.get("selected") is True for v in validations):
                    raise ValueError("static checkpoint was not selected by recorded validation AP")
                files["validation"] = str(validation_path)
            if required_stage == "dynamic":
                dyn_path = path.parent / "dynamic_normalizer.json"
                dynamic_normalizer = json.loads(dyn_path.read_text())
                if dynamic_normalizer != checkpoint["dynamic_normalizer"]:
                    raise ValueError("exported dynamic normalizer differs from checkpoint")
                if (dynamic_normalizer.get("source_split") != "train"
                        or not set(dynamic_normalizer["catalog_ids"]) <= train_catalogs):
                    raise ValueError("dynamic normalizer exceeds the declared train subset")
                files["dynamic_normalizer"] = str(dyn_path)
                samples_path = Path(checkpoint["config"]["samples_path"])
                if checkpoint["identity"].get("samples_sha256") != file_sha256(samples_path):
                    raise ValueError("dynamic sample corpus differs from training identity")
                with samples_path.open() as stream:
                    for line in stream:
                        if line.strip():
                            sample_scope = str(json.loads(line).get("experiment_scope", "")).lower()
                            if "smoke" in sample_scope or "dev" in sample_scope:
                                raise ValueError("development/smoke dynamic samples cannot enter formal evaluation")
                files["dynamic_samples"] = str(samples_path)
                selection_path = path.parent / "validation_selection.json"
                selection = validate_dynamic_selection(selection_path, document, method, int(seed), file_sha256(path))
                files["validation_selection"] = str(selection_path)
                files["validation_records"] = selection["records_path"]
            artifacts[method][seed] = {key: {"path": value, "sha256": file_sha256(value)}
                                       for key, value in files.items()}
            model_hashes[method][seed] = file_sha256(path)

    calibration = json.loads(Path(calibration_path).read_text())
    env = environment(instance_id)
    code_hash = source_hash()
    if (calibration.get("complete") is not True or calibration.get("selection") != "final"
            or calibration.get("code_hash") != code_hash
            or calibration.get("environment_hash") != env["environment_hash"]
            or calibration.get("freeze_hash") != document["freeze"]["freeze_hash"]):
        raise ValueError("test calibration incomplete or code/environment/protocol changed")
    expected_ids = set(document["splits"]["splits"]["test"])
    expected_ids.add(document["protocol"]["acceptance"]["application"]["catalog_id"])
    if set(calibration["budgets"]) != expected_ids:
        raise ValueError("final calibration must cover exactly all 90 frozen test catalogs and known application")
    for value in calibration["budgets"].values():
        if not math.isclose(value["budget_s"], calibrated_budget(value["calibration_s"]), abs_tol=1e-8):
            raise ValueError("calibration budget formula mismatch")
    raw = Path(pilot_records_path).read_text()
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    gate = evaluate_pilot(rows, document)
    if not gate["pass"]:
        raise ValueError("measured pilot gate did not pass; expansion/test freeze refused")
    if any(row.get("code_hash") != code_hash or row.get("environment_hash") != env["environment_hash"] for row in rows):
        raise ValueError("pilot code/environment changed; remeasure before test freeze")
    identity = {"models": model_hashes, "code_hash": code_hash,
                "calibration_hash": file_sha256(calibration_path),
                "protocol_hash": document["freeze"]["freeze_hash"],
                "environment_hash": env["environment_hash"],
                "inference_device": inference_device}
    implementation = Path(directory).resolve() / "IMPLEMENTATION-v1.json"
    expected_implementation_hash = (implementation.parent / "IMPLEMENTATION-v1.sha256").read_text().split()[0]
    if file_sha256(implementation) != expected_implementation_hash:
        raise ValueError("registered implementation metadata hash mismatch")
    evidence = {"dependency_lock": Path(dependency_lock), "pilot_records": Path(pilot_records_path),
                "calibration": Path(calibration_path), "implementation": implementation}
    result = {"version": PROTOCOL_VERSION, "identity": identity, "artifacts": artifacts,
              "environment": env, "dependency_lock_sha256": file_sha256(dependency_lock),
              "pilot_records_sha256": file_sha256(pilot_records_path), "pilot_gate": gate,
              "evidence": {key: {"path": str(value.resolve()), "sha256": file_sha256(value)}
                           for key, value in evidence.items()},
              "pilot_gate_passed": True, "model_selection_scope": "validation only; test not yet evaluated"}
    result["runtime_freeze_hash"] = canonical_sha256(result)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def load_runtime_freeze(path: Path | str, protocol_directory: Path | str) -> dict:
    """Verify the freeze and every referenced artifact before any test solve."""
    result = json.loads(Path(path).read_text())
    payload = {k: v for k, v in result.items() if k != "runtime_freeze_hash"}
    if result.get("runtime_freeze_hash") != canonical_sha256(payload):
        raise ValueError("runtime freeze hash mismatch")
    frozen = load_frozen_protocol(protocol_directory)
    if result["identity"]["protocol_hash"] != frozen["freeze"]["freeze_hash"]:
        raise ValueError("runtime protocol mismatch")
    for seeds in result["artifacts"].values():
        for files in seeds.values():
            for artifact in files.values():
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("runtime model artifact hash mismatch")
    for artifact in result["evidence"].values():
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("runtime evidence hash mismatch")
    from .evaluation import evaluate_pilot
    raw = Path(result["evidence"]["pilot_records"]["path"]).read_text()
    rows = json.loads(raw) if raw.lstrip().startswith("[") else [json.loads(line) for line in raw.splitlines() if line.strip()]
    gate = evaluate_pilot(rows, frozen)
    if not gate["pass"] or gate != result["pilot_gate"] or result.get("pilot_gate_passed") is not True:
        raise ValueError("runtime pilot evidence mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-splits")
    build.add_argument("--specs", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    for command in ("freeze", "verify"):
        sub.add_parser(command).add_argument("directory", type=Path)
    runtime = sub.add_parser("freeze-runtime")
    runtime.add_argument("--protocol", required=True, type=Path)
    runtime.add_argument("--models", required=True, type=Path)
    runtime.add_argument("--calibration", required=True, type=Path)
    runtime.add_argument("--pilot-records", required=True, type=Path)
    runtime.add_argument("--dependency-lock", required=True, type=Path)
    runtime.add_argument("--instance-id", required=True)
    runtime.add_argument("--inference-device", choices=("cpu", "cuda"), default="cpu")
    runtime.add_argument("--output", required=True, type=Path)
    verify_runtime = sub.add_parser("verify-runtime")
    verify_runtime.add_argument("path", type=Path)
    verify_runtime.add_argument("--protocol", required=True, type=Path)
    select = sub.add_parser("select-dynamic", help="record a checkpoint choice using measured validation planner pairs")
    select.add_argument("--protocol", required=True, type=Path)
    select.add_argument("--records", required=True, type=Path)
    select.add_argument("--checkpoint", required=True, type=Path)
    select.add_argument("--method", required=True, choices=("neural", "dynamic_random_init"))
    select.add_argument("--seed", required=True, type=int, choices=SEEDS)
    select.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build-splits":
        output = build_splits(json.loads(args.specs.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(output, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output)}))
    elif args.command == "freeze-runtime":
        result = freeze_runtime(directory=args.protocol, models_path=args.models,
                                calibration_path=args.calibration, pilot_records_path=args.pilot_records,
                                dependency_lock=args.dependency_lock, instance_id=args.instance_id,
                                output=args.output, inference_device=args.inference_device)
        print(json.dumps({"output": str(args.output), "runtime_freeze_hash": result["runtime_freeze_hash"]}))
    elif args.command == "verify-runtime":
        result = load_runtime_freeze(args.path, args.protocol)
        print(json.dumps({"runtime_freeze_hash": result["runtime_freeze_hash"], "verified": True}))
    elif args.command == "select-dynamic":
        selection = {"checkpoint_sha256": file_sha256(args.checkpoint),
                     "records_path": str(args.records.resolve()), "criterion": "scene_mean_relative_S_vs_rule"}
        selection["validation_summary"] = validate_dynamic_selection(selection, load_frozen_protocol(args.protocol),
                                                                      args.method, args.seed, selection["checkpoint_sha256"])
        with args.output.open("x") as stream:
            json.dump(selection, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(selection))
    elif args.command == "freeze":
        print(json.dumps(freeze(args.directory)))
    else:
        print(json.dumps(load_frozen_protocol(args.directory)["freeze"]))


if __name__ == "__main__":
    main()
