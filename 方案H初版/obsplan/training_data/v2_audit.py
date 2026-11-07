"""Stream-audit the frozen v0.2 corpus and its saved one-second evidence.

Only one catalog's large arrays are retained at a time. Orbital identities use
an on-disk temporary SQLite index, including the independently supplied v0.1
reference. This audit does not rerun Skyfield or infer optimality from legality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np

from ..visibility import Windows
from .audit import (AuditError, audit_files, audit_orbits, audit_scene, audit_windows,
                    group_summary, npz, read_json, require, same, sha256)
from .loader import DatasetReader

VERSION = "synthetic-v0.2"
EXPECTED_SPLITS = {"train": 420, "validation": 90, "test": 90}


class SQLiteOrbitOwner:
    """Small mapping interface consumed by the shared orbital audit."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("CREATE TABLE owner (hash BLOB PRIMARY KEY, catalog TEXT, target INTEGER) WITHOUT ROWID")
        self.last_key = None
        self.last_value = None
        self.count = 0

    def __contains__(self, key):
        row = self.db.execute("SELECT catalog,target FROM owner WHERE hash=?", (bytes.fromhex(key),)).fetchone()
        self.last_key, self.last_value = key, row
        return row is not None

    def get(self, key, default=None):
        if key != self.last_key:
            self.__contains__(key)
        return self.last_value if self.last_value is not None else default

    def __setitem__(self, key, value):
        try:
            self.db.execute("INSERT INTO owner VALUES (?,?,?)", (bytes.fromhex(key), value[0], int(value[1])))
        except sqlite3.IntegrityError as exc:
            raise AuditError(f"duplicate orbital parameters: {value}") from exc
        self.last_key, self.last_value = key, value
        self.count += 1
        if self.count % 10000 == 0:
            self.db.commit()

    def close(self):
        self.db.close()


def _canonical_specs():
    from .v2_sampling import specs
    return specs()


def _sampling_report(rows):
    from .v2_sampling import sampling_report
    return sampling_report(rows)


def audit_spec_contract(manifest: dict) -> tuple[list[dict], dict]:
    require(manifest.get("dataset_version") == VERSION, "not a v0.2 dataset")
    identity = manifest["identity"]
    require(type(identity.get("pilot")) is bool, "pilot identity flag must be explicit boolean")
    require(identity.get("dataset_version", VERSION) == VERSION, "identity version mismatch")
    require(identity.get("window_budget") == 12, "v0.2 window budget differs from the frozen protocol")
    rows = identity.get("catalog_specs")
    require(isinstance(rows, list) and bool(rows), "missing frozen catalog specs")
    ids = [r["catalog_id"] for r in rows]
    require(len(set(ids)) == len(ids), "duplicate catalog spec")
    require(all(re.fullmatch(r"v2-cat-\d{3}", x) for x in ids), "noncanonical v0.2 catalog ID")
    pilot = identity.get("pilot") is True
    if "specs_sha256" in identity:
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False).encode()).hexdigest()
        require(identity["specs_sha256"] == digest, "frozen spec fingerprint differs")
    for row in rows:
        require(row["split"] in EXPECTED_SPLITS, "unknown spec split")
        require(type(row["n_stations"]) is int and 4 <= row["n_stations"] <= 16, "invalid spec station count")
        require(type(row["n_targets"]) is int and (1 if pilot else 1000) <= row["n_targets"] <= 16000,
                "invalid spec target count")
    if not pilot:
        require(rows == _canonical_specs(), "catalog specs differ from the frozen v0.2 design")
        require(len(rows) == 600 and Counter(r["split"] for r in rows) == EXPECTED_SPLITS,
                "v0.2 requires exactly 600 catalogs with 420/90/90 splits")
        require(Counter(r["distribution"] for r in rows) == {"nominal_gaussian": 540, "stress": 60},
                "v0.2 core/stress counts changed")
        require(not any(r["distribution"] == "stress" and r["split"] == "train" for r in rows),
                "stress catalogs leaked into training")
    report = _sampling_report(rows)
    saved = manifest.get("sampling_report")
    require(isinstance(saved, dict), "missing recorded sampling report")
    require(saved == report, "recorded sampling report differs from the frozen specs")
    if not pilot:
        require(report.get("passed") is True, "sampling distribution checks failed")
    return rows, report


def load_v1_reference(root: Path | None, manifest: dict, owners: SQLiteOrbitOwner, *, pilot: bool) -> dict:
    if root is None:
        require(pilot, "formal v0.2 audit requires --v1-root for orbital non-reuse verification")
        return dict(checked=False, reason="pilot without external reference")
    root = root.resolve()
    path = root / "manifest.json"
    reference = read_json(path)
    require(reference.get("dataset_version") == "synthetic-v0.1" and reference.get("status") == "complete"
            and reference.get("identity", {}).get("pilot") is False, "reference is not complete formal v0.1")
    manifest_hash = sha256(path)
    expected = manifest["identity"].get("v1_manifest_sha256")
    require(expected == manifest_hash, "v0.1 reference manifest differs from the frozen identity")
    entries = {r["path"]: r for r in reference["files"]}
    require(len(entries) == len(reference["files"]), "duplicate reference file record")
    catalogs = reference["catalogs"]
    require(len(catalogs) == 30 and len({c["catalog_id"] for c in catalogs}) == 30,
            "v0.1 reference catalog inventory differs")
    targets = 0
    for cat in catalogs:
        relative = f"catalogs/{cat['catalog_id']}/targets.npz"
        entry = entries.get(relative)
        full = root / relative
        require(entry is not None and full.resolve().is_relative_to(root) and full.is_file(),
                "missing or escaping v0.1 target table")
        require(full.stat().st_size == entry["bytes"] and sha256(full) == entry["sha256"],
                "v0.1 target table checksum mismatch")
        table = npz(full)
        audit_orbits(table, {**cat, "catalog_id": "v0.1/"+cat["catalog_id"]}, owners)
        targets += cat["n_targets"]
        del table
    return dict(checked=True, manifest_sha256=manifest_hash, catalogs=len(catalogs), targets=targets,
                identity="SGP4 mean elements, epoch and drag; identity fields excluded")


def audit_plan_provenance(path: Path) -> int:
    """Bind both published plans to original insertion rows, including repairs."""
    grid, final = npz(path / "grid_plan.npz"), npz(path / "plan.npz")
    reports = read_json(path / "validation.json")
    n_original = reports["grid"]["original_tasks"]
    require(reports["final"]["original_tasks"] == n_original, "grid/final original task count differs")
    original_grid = npz(path / "grid_before_repair.npz") if reports["grid"]["removed_tasks"] else grid
    require(len(original_grid["station"]) == n_original, "original grid provenance length differs")
    with np.load(path / "actions.npz", allow_pickle=False) as actions:
        cw, cm = actions["candidate_window"], actions["candidate_duration_cells"]
    with np.load(path / "pool_windows.npz", allow_pickle=False) as pool:
        pool_arrays = {k: pool[k] for k in ("station", "target", "k0", "k1")}
    for label, plan in (("grid", grid), ("final", final)):
        n = len(plan["station"])
        rows, candidates = plan.get("source_grid_row"), plan.get("candidate_id")
        require(rows is not None and rows.shape == (n,) and np.issubdtype(rows.dtype, np.integer)
                and len(np.unique(rows)) == n and ((rows >= 0) & (rows < n_original)).all(),
                f"{label}: malformed original source_grid_row")
        require(candidates is not None and candidates.shape == (n,)
                and np.issubdtype(candidates.dtype, np.integer)
                and ((candidates >= 0) & (candidates < len(cw))).all(), f"{label}: invalid provenance candidate_id")
        if reports[label]["removed_tasks"]:
            invalid = npz(path / f"{label}_before_repair_geometry.npz")["invalid_mask"]
            expected_rows = np.flatnonzero(~invalid)
        else:
            expected_rows = np.arange(n_original)
        require(np.array_equal(rows, expected_rows), f"{label}: source_grid_row does not match retained repair mask")
        windows = cw[candidates]
        require(((windows >= 0) & (windows < len(pool_arrays["station"]))).all(), "provenance window out of range")
        for key in ("station", "target"):
            require(np.array_equal(plan[key], original_grid[key][rows])
                    and np.array_equal(original_grid[key][rows], pool_arrays[key][windows]),
                    f"{label}: original task identity differs from candidate")
        require(np.array_equal(original_grid["dur_s"][rows], cm[candidates].astype(np.int64)*30)
                and (original_grid["start_s"][rows] >= pool_arrays["k0"][windows]*30).all()
                and (original_grid["start_s"][rows]+original_grid["dur_s"][rows]
                     <= pool_arrays["k1"][windows]*30).all(), f"{label}: candidate does not generate original grid task")
    _, gi, fi = np.intersect1d(grid["source_grid_row"], final["source_grid_row"], return_indices=True)
    require(np.array_equal(grid["candidate_id"][gi], final["candidate_id"][fi]),
            "shared original task has inconsistent candidate identity")
    return len(final["station"])


def audit_failure_inventory(root: Path, manifest: dict, spec_map: dict[str, dict], events: list[dict]) -> dict:
    """Failed attempts are immutable payload too, never silently omitted."""
    files = [r["path"] for r in manifest["files"] if r["path"].startswith("failures/")]
    attempts = set()
    for name in files:
        parts = Path(name).parts
        require(len(parts) >= 4 and parts[1] in spec_map and re.fullmatch(r"attempt-\d+", parts[2]),
                f"noncanonical retained failure path: {name}")
        attempts.add((parts[1], parts[2]))
    ledger = manifest.get("failures", [])
    require(isinstance(ledger, list), "failure ledger must be a list")
    by_path = {}
    identity_hash = hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True,
                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    for record in ledger:
        path = record.get("path")
        require(isinstance(path, str) and path not in by_path, "missing or duplicate failure ledger path")
        parts = Path(path).parts
        require(len(parts) == 3 and parts[0] == "failures" and (parts[1], parts[2]) in attempts,
                "failure ledger refers to a missing attempt")
        original = read_json(root / path / "failure.json")
        require(original == {k:v for k,v in record.items() if k != "path"}, "retained failure report differs from ledger")
        spec = spec_map[parts[1]]
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False).encode()).hexdigest()
        require(original.get("catalog_id") == parts[1] and original.get("scene_id") == parts[1]+"-v0"
                and original.get("seed") == spec["seed"] and original.get("spec_sha256") == spec_hash
                and original.get("identity_sha256") == identity_hash
                and original.get("attempt") == int(parts[2].split("-")[1]), "retained failure identity differs")
        by_path[path] = record
    require(set(by_path) == {f"failures/{c}/{a}" for c,a in attempts}, "retained attempts are missing from failure ledger")
    logged = set()
    for event in events:
        if event.get("path") in by_path:
            record = by_path[event["path"]]
            require(all(event.get(k) == v for k,v in record.items() if k != "time_utc"),
                    "failure event differs from retained ledger")
            logged.add(event["path"])
    return dict(retained_attempts=len(attempts), retained_files=len(files), recorded_failures=len(ledger),
                recorded_timeouts=sum(r.get("type") in ("TimeoutError", "TimeoutExpired") for r in ledger),
                ledger_entries_without_event=len(set(by_path)-logged),
                attempts=[dict(catalog_id=c, attempt=a) for c, a in sorted(attempts)])


def audit_dataset(root: str | Path, *, allow_pilot: bool = False, v1_root: str | Path | None = None) -> dict:
    root = Path(root).resolve()
    reader = DatasetReader(root)
    manifest = reader.manifest
    require(manifest.get("status") == "complete", "dataset build is not complete")
    pilot = manifest["identity"].get("pilot") is True
    require(not pilot or allow_pilot, "pilot needs explicit --allow-pilot; it is not the full dataset")
    specs, sampling = audit_spec_contract(manifest)
    file_result = audit_files(root, manifest)
    spec_map = {s["catalog_id"]: s for s in specs}
    catalogs = manifest.get("catalogs", [])
    require(len(catalogs) == len(specs) == len(reader.scenes), "every catalog must have exactly one scene")
    require(len({c["catalog_id"] for c in catalogs}) == len(catalogs)
            and {c["catalog_id"] for c in catalogs} == spec_map.keys(), "catalog/spec inventory differs")
    scenes = {}
    for scene in reader.scenes:
        cid = scene["catalog_id"]
        require(cid not in scenes and cid in spec_map, "duplicate or unrecognized scene catalog")
        require(scene["scene_id"] == f"{cid}-v0" and scene["path"] == f"scenes/{cid}-v0"
                and scene.get("variant") == 0, "noncanonical v0.2 scene identity")
        scenes[cid] = scene
    repairs = provenance = 0
    with tempfile.TemporaryDirectory(prefix="obsplan-v2-orbit-audit-") as temporary:
        owners = SQLiteOrbitOwner(Path(temporary) / "orbits.sqlite")
        try:
            reference = load_v1_reference(Path(v1_root) if v1_root is not None else None,
                                          manifest, owners, pilot=pilot)
            for cat in sorted(catalogs, key=lambda c: c["catalog_id"]):
                cid, spec = cat["catalog_id"], spec_map[cat["catalog_id"]]
                same(cat, spec, f"{cid}.spec")
                cpath = root / "catalogs" / cid
                require(read_json(cpath / "catalog.json") == cat, f"{cid}: catalog manifest mismatch")
                table = npz(cpath / "targets.npz")
                audit_orbits(table, cat, owners, n_stations=spec["n_stations"])
                windows = Windows.load(cpath / "windows.npz", spec["n_stations"])
                audit_windows(windows, spec["n_targets"], spec["n_stations"], cid)
                require(cat["n_windows"] == len(windows.k0) and cat["sgp4_failed"] == 0,
                        f"{cid}: window metadata differs")
                record = scenes[cid]
                require(record["n_stations"] == spec["n_stations"]
                        and record["base_station_indices"] == list(range(spec["n_stations"])),
                        f"{cid}: v0.2 scene must retain its full sampled station network")
                repairs += audit_scene(root, reader, record, cat, table, windows,
                                       catalog_n_stations=spec["n_stations"])
                provenance += audit_plan_provenance(root / record["path"])
                del table, windows
        finally:
            owners.close()
    counts = dict(catalogs=len(catalogs), scenes=len(reader.scenes),
                  unique_synthetic_targets=sum(c["n_targets"] for c in catalogs),
                  target_instances=sum(s["n_targets"] for s in reader.scenes),
                  candidate_rows=sum(s["candidate_rows"] for s in reader.scenes),
                  grid_tasks=sum(s["grid_tasks"] for s in reader.scenes),
                  final_tasks=sum(s["final_tasks"] for s in reader.scenes), repaired_tasks=repairs)
    same(manifest["summary"], counts, "manifest.summary")
    require(counts["candidate_rows"] > 0, "no candidate rows")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    kinds = Counter(e["kind"] for e in events)
    require(kinds["build_complete"] >= 1, "missing build_complete event")
    failure_events = [e for e in events if "fail" in e["kind"] or "interrupt" in e["kind"]
                      or e.get("type") in ("TimeoutError", "TimeoutExpired", "KeyboardInterrupt")]
    retained = audit_failure_inventory(root, manifest, spec_map, events)
    groups = {}
    for key in ("split", "distribution", "size_bin", "stress_type"):
        grouped = defaultdict(list)
        for row in reader.scenes:
            grouped[str(spec_map[row["catalog_id"]].get(key))].append(row)
        groups[key] = {name: group_summary(rows) for name, rows in grouped.items()}
    starts = [datetime.fromisoformat(e["time_utc"]) for e in events if e["kind"] == "build_start"]
    ends = [datetime.fromisoformat(e["time_utc"]) for e in events if e["kind"] == "build_complete"]
    timings = dict(catalog_geometry_wall_s=sum(c["geometry_wall_s"] for c in catalogs),
                   teacher_wall_s=sum(s["teacher_timings"]["total_s"] for s in reader.scenes),
                   validation_wall_s=sum(s["validation_wall_s"] for s in reader.scenes),
                   scene_wall_s=sum(s["scene_wall_s"] for s in reader.scenes),
                   elapsed_first_start_to_last_complete_s=(max(ends)-min(starts)).total_seconds() if starts else None,
                   interpretation="production costs; concurrent stage sums overlap and are not algorithm speed rankings")
    return dict(passed=True, dataset_version=VERSION, pilot=pilot, checksums=file_result, counts=counts,
                sampling_report=sampling, v1_reference=reference,
                plan_provenance=dict(passed=True, matched_final_tasks=provenance),
                quality=group_summary(reader.scenes)["quality"], groups=groups, timings=timings,
                events=dict(counts=dict(kinds), failures=retained["recorded_failures"],
                            failure_events=len(failure_events), timeouts=retained["recorded_timeouts"],
                            scenes_with_repairs=sum(s["repairs"] > 0 for s in reader.scenes), repaired_tasks=repairs,
                            **retained),
                evidence_scope="saved one-second grid/final evidence checked; Skyfield was not rerun",
                split_isolation="frozen catalog specs plus full orbital hashes, including the v0.1 reference",
                memory_policy="one catalog at a time; exact orbital identities in temporary SQLite")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--v1-root", type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None:
        require(not args.output.resolve().is_relative_to(args.root.resolve()), "audit output must be outside dataset root")
        require(not args.output.exists(), "audit output already exists")
    result = audit_dataset(args.root, allow_pilot=args.allow_pilot, v1_root=args.v1_root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
