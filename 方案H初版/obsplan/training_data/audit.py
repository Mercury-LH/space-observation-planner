"""Audit published synthetic inputs, labels, checksums and saved validation evidence.

This audit does not rerun Skyfield.  The producer independently validates every
grid/final plan at one-second resolution; here we check the saved evidence and
its consistency with the released plans and files.  It is not a second physical
simulation and is not proof that every candidate window is feasible at 1 second.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from ..config import Config
from ..data import load, tle_checksum_ok
from ..schedule import Schedule, evaluate
from ..visibility import Windows
from .loader import DatasetReader


class AuditError(ValueError):
    """The released corpus does not satisfy its recorded contract."""


def require(condition, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def same(actual, expected, name: str) -> None:
    """Numerically compare recomputed metrics, including their nested arrays."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{name}: expected object")
        for key, value in expected.items():
            require(key in actual, f"{name}: missing {key}")
            same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, (list, tuple, np.ndarray, int, float, np.number)):
        try:
            a, b = np.asarray(actual, float), np.asarray(expected, float)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"{name}: expected finite numeric value") from exc
        require(a.shape == b.shape and np.isfinite(a).all()
                and np.allclose(a, b, rtol=1e-10, atol=1e-9), f"{name}: value mismatch")
    else:
        require(actual == expected, f"{name}: value mismatch")


def audit_files(root: Path, manifest: dict) -> dict:
    entries = manifest.get("files")
    require(isinstance(entries, list) and bool(entries), "manifest.files is empty or absent")
    seen, total_bytes = set(), 0
    for entry in entries:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        require(not relative.is_absolute() and ".." not in relative.parts
                and path.is_relative_to(root), "manifest file path escapes dataset")
        require(entry["path"] not in seen, f"duplicate file entry: {relative}")
        seen.add(entry["path"])
        require(path.is_file(), f"missing manifest file: {relative}")
        require(path.stat().st_size == entry["bytes"], f"byte size mismatch: {relative}")
        require(sha256(path) == entry["sha256"], f"SHA-256 mismatch: {relative}")
        total_bytes += path.stat().st_size
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and p.name not in ("manifest.json", "events.jsonl") and ".cache" not in p.parts}
    require(actual == seen, f"incomplete file manifest: unlisted={sorted(actual-seen)[:5]}, "
            f"absent={sorted(seen-actual)[:5]}")
    return {"verified_files": len(seen), "verified_bytes": total_bytes,
            "manifest_sha256": sha256(root / "manifest.json"),
            "events_sha256": sha256(root / "events.jsonl"),
            "manifest_exclusions": ["manifest.json", "events.jsonl", ".cache/**"]}


def audit_windows(win: Windows, n_targets: int, n_stations: int, name: str) -> None:
    arrays = (win.station, win.target, win.k0, win.k1)
    n = len(win.k0)
    require(all(a.ndim == 1 and len(a) == n and np.issubdtype(a.dtype, np.integer)
                for a in arrays), f"{name}: malformed window arrays")
    require(win.n_samples == 17281, f"{name}: not the frozen six-day/30-second horizon")
    require(((win.station >= 0) & (win.station < n_stations)).all(), f"{name}: station out of range")
    require(((win.target >= 0) & (win.target < n_targets)).all(), f"{name}: target out of range")
    require(((win.k0 >= 0) & (win.k1 < win.n_samples) & (win.k1-win.k0 >= 4)).all(),
            f"{name}: invalid window interval")
    require(win.sgp4_failed.shape == (n_targets,) and win.sgp4_failed.dtype == np.dtype(bool)
            and not win.sgp4_failed.any(), f"{name}: propagation failures or malformed flags")
    require(len(win.dark) == n_stations, f"{name}: wrong dark interval count")
    for runs in win.dark:
        previous_end = -1
        for a, b in runs:
            require(0 <= a < b < win.n_samples and a > previous_end, f"{name}: invalid dark intervals")
            previous_end = b


def audit_orbits(table: dict, cat: dict, orbit_owner: dict) -> None:
    n, name = cat["n_targets"], cat["catalog_id"]
    for key in ("satnum", "target_id", "line1", "line2", "family", "names"):
        require(table[key].shape == (n,), f"{name}: wrong target table shape for {key}")
    require(np.array_equal(table["target_id"], np.arange(n)), f"{name}: target IDs are not zero based")
    require(len(np.unique(table["satnum"])) == n, f"{name}: duplicate synthetic target number")
    require(table["stations"].shape == (10, 3) and np.isfinite(table["stations"]).all(),
            f"{name}: invalid station table")
    require(((table["stations"][:, 0] >= -180) & (table["stations"][:, 0] < 180)).all()
            and (np.abs(table["stations"][:, 1]) <= 90).all(), f"{name}: invalid station coordinates")
    for i, (line1, line2) in enumerate(zip(table["line1"], table["line2"])):
        require(len(line1) == len(line2) == 69 and line1.startswith("1 ") and line2.startswith("2 ")
                and tle_checksum_ok(line1) and tle_checksum_ok(line2), f"{name}: TLE checksum/format error at {i}")
        sat = Satrec.twoline2rv(str(line1), str(line2))
        require(sat.satnum == int(table["satnum"][i]), f"{name}: TLE number mismatch at {i}")
        values = (sat.no_kozai, sat.ecco, sat.inclo, sat.nodeo, sat.argpo, sat.mo,
                  sat.bstar, sat.ndot, sat.nddot, sat.jdsatepoch, sat.jdsatepochF)
        require(np.isfinite(values).all() and sat.no_kozai > 0 and 0 <= sat.ecco < 1
                and 0 <= sat.inclo <= np.pi, f"{name}: invalid mean elements at {i}")
        perigee = (398600.8 / (sat.no_kozai / 60)**2)**(1/3) * (1-sat.ecco) - 6378.135
        require(perigee >= 450, f"{name}: mean-element perigee below 450 km at {i}")
        error, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF)
        require(error == 0 and np.isfinite([*r, *v]).all(), f"{name}: epoch propagation failure at {i}")
        # Only dynamical parameters: excludes satnum, names and launch designator.
        key = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
        require(key not in orbit_owner, f"duplicate orbital parameters: {name}:{i} and {orbit_owner.get(key)}")
        orbit_owner[key] = (name, i)


def load_plan(path: Path, cfg: Config, ds, name: str) -> tuple[Schedule, dict]:
    arrays = npz(path)
    keys = ("station", "target", "start_s", "dur_s")
    require(all(key in arrays for key in keys), f"{name}: missing plan arrays")
    n = len(arrays["station"])
    require(n > 0 and all(arrays[k].shape == (n,) and np.issubdtype(arrays[k].dtype, np.integer)
                         for k in keys), f"{name}: empty or malformed plan")
    sch = Schedule(*(arrays[k] for k in keys))
    require(((sch.station >= 0) & (sch.station < ds.n_stations)).all()
            and ((sch.target >= 0) & (sch.target < ds.n_targets)).all(), f"{name}: invalid task IDs")
    require(((sch.start_s >= 0) & (sch.start_s+sch.dur_s <= cfg.days*86400)
             & (sch.dur_s >= 120) & (sch.dur_s <= 240)).all(), f"{name}: invalid task times")
    for ids, gap, label in [(sch.station, 30, "station switch"), (sch.target, 48*3600, "target revisit")]:
        order = np.lexsort((sch.start_s, ids))
        shared = ids[order[:-1]] == ids[order[1:]]
        require((sch.start_s[order[1:]][shared] >=
                 (sch.start_s[order[:-1]]+sch.dur_s[order[:-1]]+gap)[shared]).all(),
                f"{name}: {label} constraint violated")
    return sch, arrays


def audit_validation(path: Path, name: str, plan: Schedule, report: dict, cfg: Config, ds) -> int:
    tag = f"{path.name}/{name}"
    require(report.get("legal") is True and report.get("geometry_checked") is True
            and report.get("step_s") == 1.0 and report.get("violations") == {},
            f"{tag}: missing successful independent one-second evidence")
    require(report.get("tasks") == len(plan), f"{tag}: validation task count mismatch")
    evidence = npz(path / f"{name}_geometry.npz")
    values, invalid = evidence["per_task"], evidence["invalid_mask"]
    require(values.shape == (len(plan), 3) and np.isfinite(values).all()
            and invalid.shape == (len(plan),) and invalid.dtype == np.dtype(bool)
            and not invalid.any(), f"{tag}: malformed or failed geometry evidence")
    require((values[:, 0] >= cfg.elev_min).all() and (values[:, 1] <= cfg.elev_max).all()
            and (values[:, 0] <= values[:, 1]).all() and (values[:, 2] < cfg.sun_max).all(),
            f"{tag}: geometry extrema violate the rules")
    for key, expected in [("min_elev_margin", values[:, 0].min()-cfg.elev_min),
                          ("max_elev_margin", cfg.elev_max-values[:, 1].max()),
                          ("sun_margin", cfg.sun_max-values[:, 2].max())]:
        same(report.get(key), expected, f"{tag}.{key}")
    removed = report.get("removed_tasks", -1)
    require(isinstance(removed, int) and removed >= 0
            and report.get("original_tasks") == len(plan)+removed, f"{tag}: repair accounting mismatch")
    attempts = report.get("failed_attempts")
    require(isinstance(attempts, list) and len(attempts) == int(removed > 0),
            f"{tag}: missing retained repair report")
    if removed:
        for suffix in (".npz", "_validation.json", "_geometry.npz"):
            require((path / f"{name}_before_repair{suffix}").is_file(), f"{tag}: missing pre-repair evidence")
        before, _ = load_plan(path / f"{name}_before_repair.npz", cfg, ds, tag)
        require(len(before) == report["original_tasks"], f"{tag}: original task count mismatch")
        previous = read_json(path / f"{name}_before_repair_validation.json")
        require(previous == attempts[0] and previous.get("legal") is False
                and previous.get("geometry_checked") is True and bool(previous.get("violations")),
                f"{tag}: retained failed report mismatch")
        old_geometry = npz(path / f"{name}_before_repair_geometry.npz")
        old_invalid = old_geometry["invalid_mask"]
        require(old_geometry["per_task"].shape == (len(before), 3)
                and old_invalid.dtype == np.dtype(bool) and old_invalid.shape == (len(before),)
                and int(old_invalid.sum()) == removed, f"{tag}: retained invalid mask mismatch")
        for key in ("station", "target", "start_s", "dur_s"):
            require(np.array_equal(getattr(plan, key), getattr(before, key)[~old_invalid]),
                    f"{tag}: repair did not remove exactly the independently flagged tasks")
        original = Counter(zip(before.station, before.target, before.start_s, before.dur_s))
        kept = Counter(zip(plan.station, plan.target, plan.start_s, plan.dur_s))
        require(not (kept-original), f"{tag}: repair changed tasks instead of deleting them")
    return removed


def stats(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def group_summary(records: list[dict]) -> dict:
    return {
        "scenes": len(records), "catalogs": len({r["catalog_id"] for r in records}),
        "target_instances": sum(r["n_targets"] for r in records),
        "candidate_rows": sum(r["candidate_rows"] for r in records),
        "grid_tasks": sum(r["grid_tasks"] for r in records),
        "final_tasks": sum(r["final_tasks"] for r in records),
        "quality": {key: stats([r["metrics"][key] for r in records])
                    for key in ("covered", "coverage_rate", "score", "combined")},
    }


def audit_dataset(root: str | Path, *, allow_pilot: bool = False) -> dict:
    root = Path(root).resolve()
    reader = DatasetReader(root)
    manifest = reader.manifest
    require(manifest.get("status") == "complete", "dataset build is not complete")
    pilot = manifest["identity"].get("pilot", False)
    require(not pilot or allow_pilot, "pilot needs explicit --allow-pilot; it is not the full dataset")
    file_result = audit_files(root, manifest)
    require(manifest["identity"]["window_budget"] == 12, "window budget differs from frozen protocol")
    catalogs, orbit_owner, catalog_records = {}, {}, {}
    require(bool(manifest.get("catalogs")), "no catalogs")
    for cat in manifest["catalogs"]:
        cat_id = cat["catalog_id"]
        require(cat_id not in catalogs, f"duplicate catalog: {cat_id}")
        path = root / "catalogs" / cat_id
        require(read_json(path / "catalog.json") == cat, f"{cat_id}: catalog manifest mismatch")
        table = npz(path / "targets.npz")
        audit_orbits(table, cat, orbit_owner)
        windows = Windows.load(path / "windows.npz", 10)
        audit_windows(windows, cat["n_targets"], 10, cat_id)
        require(cat["n_windows"] == len(windows.k0) and cat["sgp4_failed"] == 0,
                f"{cat_id}: window metadata mismatch")
        catalogs[cat_id] = (table, windows)
        catalog_records[cat_id] = cat
    require(bool(reader.scenes), "no scenes")
    used_catalogs, repaired_tasks = set(), 0
    for record in reader.scenes:
        scene_id, cat_id = record["scene_id"], record["catalog_id"]
        require(cat_id in catalogs, f"{scene_id}: missing catalog")
        used_catalogs.add(cat_id)
        cat = catalog_records[cat_id]
        require(record["split"] == cat["split"] and record["distribution"] == cat["distribution"],
                f"{scene_id}: catalog split/distribution mismatch")
        path = root / record["path"]
        require(read_json(path / "scene.json") == record, f"{scene_id}: scene manifest mismatch")
        cfg = Config(**record["config"]).with_(data_dir=str(path / "input"))
        for key, expected in dict(days=6., cell_s=30, obs_min_s=120, obs_max_s=240,
                                  switch_s=30, repeat_gap_h=48., use_sun=True, use_sunlit=True,
                                  elev_min=10., elev_max=85., sun_max=-6., moon_sep_deg=0.).items():
            require(getattr(cfg, key) == expected, f"{scene_id}: changed rule {key}")
        require(cfg.start_utc == cat["start_utc"], f"{scene_id}: planning epoch mismatch")
        ds = load(cfg)
        require(not ds.issues and ds.n_targets == record["n_targets"] == cat["n_targets"]
                and ds.n_stations == record["n_stations"], f"{scene_id}: native input mismatch/issues")
        table, full = catalogs[cat_id]
        for actual, expected in [(ds.satnum, table["satnum"]), (ds.line1, table["line1"]),
                                 (ds.line2, table["line2"]), (ds.names, table["names"])]:
            require(np.array_equal(actual, expected), f"{scene_id}: native orbit identity mismatch")
        stations = np.asarray(record["base_station_indices"])
        require(stations.ndim == 1 and len(stations) == ds.n_stations and len(np.unique(stations)) == len(stations)
                and np.issubdtype(stations.dtype, np.integer) and ((stations >= 0) & (stations < 10)).all(),
                f"{scene_id}: invalid station mapping")
        require(np.allclose(np.column_stack([ds.st_lon, ds.st_lat, ds.st_alt_m]), table["stations"][stations],
                            rtol=0, atol=5.1e-7), f"{scene_id}: native station coordinates differ")
        pool = Windows.load(path / "pool_windows.npz", ds.n_stations)
        audit_windows(pool, ds.n_targets, ds.n_stations, scene_id)
        source = npz(path / "pool_source_indices.npz")["catalog_window_id"]
        require(source.shape == (len(pool.k0),) and np.issubdtype(source.dtype, np.integer)
                and len(np.unique(source)) == len(source) and ((source >= 0) & (source < len(full.k0))).all(),
                f"{scene_id}: invalid catalog window mapping")
        inverse = np.full(10, -1, np.int16)
        inverse[stations] = np.arange(len(stations))
        for actual, expected in [(pool.station, inverse[full.station[source]]),
                                 (pool.target, full.target[source]), (pool.k0, full.k0[source]),
                                 (pool.k1, full.k1[source])]:
            require(np.array_equal(actual, expected), f"{scene_id}: pool/catalog window mismatch")
        require(pool.dark == [full.dark[s] for s in stations], f"{scene_id}: dark intervals mismatch")
        require(np.bincount(pool.target, minlength=ds.n_targets).max(initial=0) <= 12,
                f"{scene_id}: per-target pool budget exceeded")
        require(record["pool_windows"] == len(pool.k0)
                and record["full_variant_windows"] == int(np.isin(full.station, stations).sum()),
                f"{scene_id}: window count mismatch")
        actions = reader._load_actions(record)
        cw, cm = actions["candidate_window"], actions["candidate_duration_cells"]
        require(len(cw) == record["candidate_rows"] and (cw < len(pool.k0)).all(),
                f"{scene_id}: invalid candidate window indices")
        require((cm <= pool.k1[cw]-pool.k0[cw]).all(), f"{scene_id}: candidate duration cannot fit window")
        require(len(np.unique(cw.astype(np.int64)*9+cm)) == len(cw), f"{scene_id}: duplicate candidate rows")
        require(int(actions["grid_selected"].sum()) == record["grid_positive_candidates"],
                f"{scene_id}: positive label count mismatch")
        grid, grid_arrays = load_plan(path / "grid_plan.npz", cfg, ds, scene_id+"/grid")
        final, _ = load_plan(path / "plan.npz", cfg, ds, scene_id+"/final")
        ids = actions.get("grid_task_candidate")
        require(ids is not None and ids.shape == (len(grid),)
                and np.array_equal(ids, grid_arrays.get("candidate_id")), f"{scene_id}: missing task candidate mapping")
        w = cw[ids]
        require(np.array_equal(grid.station, pool.station[w]) and np.array_equal(grid.target, pool.target[w])
                and np.array_equal(grid.dur_s, cm[ids]*30) and (grid.start_s % 30 == 0).all()
                and (grid.start_s >= pool.k0[w]*30).all()
                and (grid.start_s+grid.dur_s <= pool.k1[w]*30).all(), f"{scene_id}: grid task outside its candidate")
        chosen = np.flatnonzero(actions["grid_selected"])
        first = actions["grid_selection_rank"][chosen]
        require(np.array_equal(actions["grid_start_s"][chosen], grid.start_s[first])
                and np.array_equal(actions["grid_duration_s"][chosen], grid.dur_s[first]),
                f"{scene_id}: first-selection times disagree with grid plan")
        same(record["metrics"], evaluate(cfg, ds, final), scene_id+".metrics")
        same(record["grid_metrics"], evaluate(cfg, ds, grid), scene_id+".grid_metrics")
        require(record["grid_tasks"] == len(grid) and record["final_tasks"] == len(final),
                f"{scene_id}: plan count mismatch")
        reports = read_json(path / "validation.json")
        repairs = audit_validation(path, "grid", grid, reports["grid"], cfg, ds)
        repairs += audit_validation(path, "final", final, reports["final"], cfg, ds)
        require(record["repairs"] == repairs and record["legal"] is True and record["geometry_checked"] is True,
                f"{scene_id}: validation metadata mismatch")
        for timing_name, timing_value in {**record["teacher_timings"],
                                          "validation_wall_s": record["validation_wall_s"],
                                          "scene_wall_s": record["scene_wall_s"]}.items():
            require(isinstance(timing_value, (int, float)) and np.isfinite(timing_value) and timing_value >= 0,
                    f"{scene_id}: invalid timing {timing_name}")
        same(record["validation_wall_s"], reports["grid"]["wall_s"]+reports["final"]["wall_s"],
             scene_id+".validation_wall_s")
        repaired_tasks += repairs
    require(used_catalogs == set(catalogs), "catalogs without scenes")
    counts = dict(catalogs=len(catalogs), scenes=len(reader.scenes),
                  unique_synthetic_targets=sum(c["n_targets"] for c in manifest["catalogs"]),
                  target_instances=sum(s["n_targets"] for s in reader.scenes),
                  candidate_rows=sum(s["candidate_rows"] for s in reader.scenes),
                  grid_tasks=sum(s["grid_tasks"] for s in reader.scenes),
                  final_tasks=sum(s["final_tasks"] for s in reader.scenes), repaired_tasks=repaired_tasks)
    same(manifest["summary"], counts, "manifest.summary")
    require(counts["candidate_rows"] > 0, "no training candidates")
    if not pilot:
        require(counts["catalogs"] >= 30 and counts["scenes"] >= 120 and counts["candidate_rows"] >= 1_000_000,
                "full-dataset size thresholds not met")
        require({c["n_targets"] for c in manifest["catalogs"]} >= {1000, 5000, 10000}, "missing required target scales")
        require(Counter(c["split"] for c in manifest["catalogs"]) == {"train": 20, "validation": 5, "test": 5},
                "frozen catalog split counts changed")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line.strip()]
    kinds = Counter(e["kind"] for e in events)
    failures = [e for e in events if "fail" in e["kind"] or e.get("type") in ("TimeoutError", "TimeoutExpired")]
    require(kinds["build_complete"] >= 1, "missing build_complete event")
    groups = {}
    for key in ("split", "n_targets", "distribution"):
        grouped = defaultdict(list)
        for scene in reader.scenes:
            grouped[str(scene[key])].append(scene)
        groups[key] = {name: group_summary(records) for name, records in grouped.items()}
    starts = [datetime.fromisoformat(e["time_utc"]) for e in events if e["kind"] == "build_start"]
    ends = [datetime.fromisoformat(e["time_utc"]) for e in events if e["kind"] == "build_complete"]
    timing = {
        "catalog_geometry_wall_s": sum(c["geometry_wall_s"] for c in manifest["catalogs"]),
        "teacher_wall_s": sum(s["teacher_timings"]["total_s"] for s in reader.scenes),
        "validation_wall_s": sum(s["validation_wall_s"] for s in reader.scenes),
        "scene_wall_s": sum(s["scene_wall_s"] for s in reader.scenes),
        "elapsed_first_start_to_last_complete_s": (max(ends)-min(starts)).total_seconds() if starts else None,
        "interpretation": "data production records; elapsed time includes interruptions; not a solver speed comparison",
    }
    return dict(passed=True, pilot=pilot, checksums=file_result, counts=counts,
                quality=group_summary(reader.scenes)["quality"], groups=groups, timings=timing,
                events=dict(counts=dict(kinds), failures=len(failures),
                            timeouts=sum(e.get("type") in ("TimeoutError", "TimeoutExpired") for e in failures),
                            scenes_with_repairs=sum(s["repairs"] > 0 for s in reader.scenes), repaired_tasks=repaired_tasks),
                evidence_scope="saved one-second grid/final validation evidence checked; Skyfield was not rerun",
                orbital_identity="SGP4 mean elements, epoch and drag; excludes satnum/name/designator",
                split_isolation="catalog IDs and orbital-parameter hashes checked; variants remain in one split")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--output", type=Path, help="new JSON outside dataset root; refuses overwrite")
    args = parser.parse_args(argv)
    if args.output is not None:
        require(not args.output.resolve().is_relative_to(args.root.resolve()),
                "audit output must be outside dataset root to preserve manifest completeness")
        require(not args.output.exists(), "audit output already exists; choose a new path")
    result = audit_dataset(args.root, allow_pilot=args.allow_pilot)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
