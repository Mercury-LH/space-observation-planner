"""Leakage-aware NumPy batches for the synthetic scheduling corpus.

``grid_selected`` means selected by the grid teacher, not physically legal.
The 20 input features are computed before insertion; these are candidate
ranking examples, not full state/action/reward/next-state transitions.  Batches
never cross a scene boundary.  Each compressed scene shard is loaded into RAM,
then sliced; the complete corpus does not need to fit into RAM.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ..famou_h_eval import FEATURES

SPLITS = ("train", "validation", "test")
ACTION_DTYPES = {
    "features": np.dtype("float32"),
    "candidate_window": np.dtype("int32"),
    "candidate_duration_cells": np.dtype("int16"),
    "rule_score_uncovered": np.dtype("float32"),
    "rule_score_covered": np.dtype("float32"),
    "grid_selected": np.dtype("bool"),
    "grid_selection_rank": np.dtype("int32"),
    "grid_start_s": np.dtype("int32"),
    "grid_duration_s": np.dtype("int32"),
}


def _split(value: str) -> str:
    if value == "val":
        value = "validation"
    if value not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {value!r}")
    return value


@dataclass(frozen=True)
class Normalizer:
    """Population mean/std fitted using train rows only; constant std becomes 1."""

    mean: np.ndarray
    scale: np.ndarray
    row_count: int
    scene_ids: tuple[str, ...]
    catalog_ids: tuple[str, ...]
    train_manifest_sha256: str

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError("features must be a finite [N, 20] array")
        return np.ascontiguousarray((x - self.mean) / self.scale, dtype=np.float32)

    def as_dict(self) -> dict:
        return {
            "schema_version": 1, "source_split": "train",
            "feature_names": FEATURES, "row_count": self.row_count,
            "scene_ids": list(self.scene_ids), "catalog_ids": list(self.catalog_ids),
            "train_manifest_sha256": self.train_manifest_sha256,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "variance_convention": "population; constant feature scale is 1",
        }

    def save(self, path: str | Path) -> None:
        """Write a new artifact; do not silently replace a previous run."""
        with Path(path).open("x", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("source_split") != "train" or obj.get("feature_names") != FEATURES:
            raise ValueError("normalizer must be fitted on train with the canonical features")
        mean, scale = np.asarray(obj["mean"], np.float64), np.asarray(obj["scale"], np.float64)
        if (mean.shape != (len(FEATURES),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or (scale <= 0).any() or obj["row_count"] <= 0):
            raise ValueError("invalid normalization statistics")
        return cls(mean, scale, int(obj["row_count"]), tuple(obj["scene_ids"]),
                   tuple(obj["catalog_ids"]), obj["train_manifest_sha256"])


class DatasetReader:
    """Load manifest-defined splits; reject catalog or scene leakage early."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("feature_names", FEATURES) != FEATURES:
            raise ValueError("manifest feature_names do not match the canonical 20 features")
        self.scenes = self.manifest.get("scenes")
        if not isinstance(self.scenes, list):
            raise ValueError("manifest.scenes must be a list")
        seen_ids, seen_paths, catalog_split = set(), set(), {}
        for record in self.scenes:
            if not isinstance(record, dict):
                raise ValueError("each scene must be an object")
            for key in ("scene_id", "catalog_id", "split", "path"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"scene requires a nonempty {key}")
            split = _split(record["split"])
            record["split"] = split
            scene_id, catalog_id = record["scene_id"], record["catalog_id"]
            path = Path(record["path"])
            resolved = (self.root / path).resolve()
            if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(self.root):
                raise ValueError(f"scene path must stay inside dataset: {record['path']}")
            if scene_id in seen_ids or resolved in seen_paths:
                raise ValueError("duplicate scene identity or scene path")
            if catalog_id in catalog_split and catalog_split[catalog_id] != split:
                raise ValueError(f"catalog leakage across splits: {catalog_id}")
            seen_ids.add(scene_id)
            seen_paths.add(resolved)
            catalog_split[catalog_id] = split
        train_records = [r for r in self.scenes if r["split"] == "train"]
        self.train_manifest_sha256 = hashlib.sha256(
            json.dumps(train_records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def records(self, split: str) -> list[dict]:
        split = _split(split)
        return [dict(r) for r in self.scenes if r["split"] == split]

    def _load_actions(self, record: dict) -> dict[str, np.ndarray]:
        path = self.root / record["path"] / "actions.npz"
        with np.load(path, allow_pickle=False) as archive:
            missing = set(ACTION_DTYPES).difference(archive.files)
            if missing:
                raise ValueError(f"{record['scene_id']}: missing action fields {sorted(missing)}")
            arrays = {name: archive[name] for name in ACTION_DTYPES}
            for name in ("grid_selected_count", "grid_task_candidate"):
                if name in archive.files:
                    arrays[name] = archive[name]
        x = arrays["features"]
        if x.ndim != 2 or x.shape[1] != len(FEATURES):
            raise ValueError(f"{record['scene_id']}: expected features shape [N, 20]")
        n = len(x)
        for name, dtype in ACTION_DTYPES.items():
            value = arrays[name]
            shape = (n, len(FEATURES)) if name == "features" else (n,)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{record['scene_id']}: wrong shape or dtype for {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"{record['scene_id']}: non-finite {name}")
        selected = arrays["grid_selected"]
        rank, start, duration = (arrays[k] for k in
                                 ("grid_selection_rank", "grid_start_s", "grid_duration_s"))
        cells = arrays["candidate_duration_cells"]
        if (arrays["candidate_window"] < 0).any() or ((cells < 4) | (cells > 8)).any():
            raise ValueError(f"{record['scene_id']}: invalid window or duration index")
        if ((rank[selected] < 0).any() or len(np.unique(rank[selected])) != int(selected.sum())
                or (start[selected] < 0).any() or (start[selected] % 30 != 0).any()
                or not np.array_equal(duration[selected], cells[selected].astype(np.int32) * 30)
                or (rank[~selected] != -1).any() or (start[~selected] != -1).any()
                or (duration[~selected] != -1).any()):
            raise ValueError(f"{record['scene_id']}: inconsistent grid teacher labels")
        if "grid_selected_count" in arrays:
            count = arrays["grid_selected_count"]
            if (count.shape != (n,) or count.dtype != np.dtype("int32")
                    or (count < 0).any() or not np.array_equal(count > 0, selected)):
                raise ValueError(f"{record['scene_id']}: inconsistent grid selection counts")
        if "grid_task_candidate" in arrays:
            task_candidate = arrays["grid_task_candidate"]
            if (task_candidate.ndim != 1 or task_candidate.dtype != np.dtype("int32")
                    or ((task_candidate < 0) | (task_candidate >= n)).any()):
                raise ValueError(f"{record['scene_id']}: invalid task-to-candidate mapping")
            ids, first, counts = np.unique(task_candidate, return_index=True, return_counts=True)
            if (not np.array_equal(ids, np.flatnonzero(selected))
                    or not np.array_equal(rank[ids], first)
                    or ("grid_selected_count" in arrays
                        and not np.array_equal(arrays["grid_selected_count"][ids], counts))):
                raise ValueError(f"{record['scene_id']}: task mapping disagrees with teacher labels")
        return arrays

    def iter_scenes(self, split: str) -> Iterator[tuple[dict, dict[str, np.ndarray]]]:
        """Yield (manifest scene record, action arrays), one compressed shard at a time."""
        for record in self.records(split):
            yield record, self._load_actions(record)

    def iter_batches(self, split: str, batch_size: int = 4096, *,
                     normalizer: Normalizer | None = None,
                     max_batches: int | None = None) -> Iterator[dict]:
        """Yield NumPy arrays ready for torch.from_numpy; never combine scenes.

        Labels are the teacher's choices.  They are not a feasibility mask and
        are not labels for every possible dynamically updated planning state.
        Remaining rows are yielded even when a scene is shorter than batch_size.
        """
        split = _split(split)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if max_batches is not None and (
                not isinstance(max_batches, int) or isinstance(max_batches, bool) or max_batches <= 0):
            raise ValueError("max_batches must be a positive integer or None")
        if normalizer is not None and normalizer.train_manifest_sha256 != self.train_manifest_sha256:
            raise ValueError("normalizer was fitted to a different train manifest")
        emitted = 0
        for record, arrays in self.iter_scenes(split):
            for offset in range(0, len(arrays["features"]), batch_size):
                batch = {name: np.ascontiguousarray(value[offset:offset + batch_size])
                         for name, value in arrays.items() if name != "grid_task_candidate"}
                if normalizer is not None:
                    batch["features"] = normalizer.transform(batch["features"])
                batch.update(scene_id=record["scene_id"], catalog_id=record["catalog_id"],
                             split=split, row_offset=offset)
                yield batch
                emitted += 1
                if max_batches is not None and emitted >= max_batches:
                    return

    def fit_normalizer(self, batch_size: int = 8192) -> Normalizer:
        """Streaming parallel-variance update over *all train rows only*."""
        count = 0
        mean = np.zeros(len(FEATURES), np.float64)
        m2 = np.zeros_like(mean)
        for batch in self.iter_batches("train", batch_size):
            x = batch["features"].astype(np.float64)
            n = len(x)
            local_mean = x.mean(axis=0)
            local_m2 = np.square(x - local_mean).sum(axis=0)
            delta = local_mean - mean
            total = count + n
            m2 += local_m2 + np.square(delta) * count * n / total
            mean += delta * n / total
            count = total
        if not count:
            raise ValueError("cannot fit a normalizer: train split has no candidate rows")
        scale = np.sqrt(np.maximum(m2 / count, 0.0))
        scale[scale == 0] = 1.0
        records = self.records("train")
        return Normalizer(mean, scale, count, tuple(r["scene_id"] for r in records),
                          tuple(sorted({r["catalog_id"] for r in records})),
                          self.train_manifest_sha256)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="extracted dataset containing manifest.json")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true", help="read one batch per split after fitting train stats")
    parser.add_argument("--normalizer-output", type=Path, help="optional NEW JSON file; default writes nothing")
    args = parser.parse_args(argv)
    reader = DatasetReader(args.root)
    normalizer = reader.fit_normalizer(args.batch_size)
    if args.normalizer_output is not None:
        normalizer.save(args.normalizer_output)
    summary = {"feature_names": FEATURES, "normalizer": normalizer.as_dict(), "splits": {},
               "label_semantics": "grid_selected is a heuristic teacher choice, not a legality label",
               "mode": "first_batch_per_split" if args.smoke else "all_batches"}
    for split in SPLITS:
        rows, selected, batches = 0, 0, 0
        for batch in reader.iter_batches(split, args.batch_size, normalizer=normalizer,
                                         max_batches=1 if args.smoke else None):
            rows += len(batch["features"])
            selected += int(batch["grid_selected"].sum())
            batches += 1
        summary["splits"][split] = {
            "scenes": len(reader.records(split)), "read_rows": rows,
            "read_selected": selected, "read_batches": batches,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
