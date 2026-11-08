"""Verify the released v0.2 manifest and every payload hash without modifying data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(root: Path, index: Path, output: Path):
    started = time.perf_counter()
    root = root.resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError('verification report must be outside the read-only dataset')
    if output.exists():
        raise FileExistsError(output)
    idx = json.loads(index.read_text())
    manifest_hash = digest(root / 'manifest.json')
    if idx['dataset_version'] != 'synthetic-v0.2' or manifest_hash != idx['manifest_sha256']:
        raise ValueError('v0.2 released manifest hash mismatch')
    manifest = json.loads((root / 'manifest.json').read_text())
    scenes = manifest['scenes']
    if Counter(r['split'] for r in scenes) != {'train': 420, 'validation': 90, 'test': 90}:
        raise ValueError('unexpected catalog split counts')
    if len({r['catalog_id'] for r in scenes}) != 600:
        raise ValueError('catalog identities are not independent')
    failures, total = [], 0
    files = manifest['files'] + [{'path': 'events.jsonl', 'sha256': idx['events_sha256']}]
    for row in files:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('manifest path escapes dataset')
        try:
            n = path.stat().st_size
            ok = ('bytes' not in row or n == row['bytes']) and digest(path) == row['sha256']
            total += n
            if not ok:
                failures.append({'path': row['path'], 'status': 'hash_or_size_mismatch'})
        except OSError as exc:
            failures.append({'path': row['path'], 'status': type(exc).__name__})
    result = dict(passed=not failures, dataset_version='synthetic-v0.2', files=len(files),
                  bytes=total, manifest_sha256=manifest_hash, index_sha256=digest(index),
                  splits=dict(Counter(r['split'] for r in scenes)), failures=failures,
                  wall_s=time.perf_counter()-started,
                  scope='released payload identity; no geometry recomputation or model evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if verify(args.root, args.index, args.output)['passed'] else 1)
