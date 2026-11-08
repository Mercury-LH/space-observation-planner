"""Reproducible static pretraining and same-state counterfactual ranking.

Examples (configuration paths are relative to the working directory):
  python -m obsplan.neural_lns.training static --config train.json --output outputs/run
  python -m obsplan.neural_lns.training static --config train.json --output outputs/run --resume outputs/run/checkpoint.pt

This module never starts paid compute or evaluates test catalogs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from ..famou_h_eval import FEATURES
from ..training_data.loader import DatasetReader
from .model import CandidateMLP
from .static_data import BalancedStream, digest, file_sha256, fit_scene_normalizer, select_records, transform


DEFAULTS = {'seed': 2026, 'steps': 10000, 'batch_size': 8192, 'batches_per_scene': 16,
            'learning_rate': 1e-3, 'weight_decay': 1e-4, 'eval_every': 500,
            'checkpoint_every': 500, 'device': 'cpu', 'threads': 1,
            'huber_weight': 0.0}


def write_json(path: Path, value, *, exclusive=True):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def append_json(path: Path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def environment() -> dict:
    packages = {}
    for name in ('torch', 'numpy', 'numba', 'skyfield', 'sgp4', 'ortools'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'packages': packages,
            'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'torch_threads': torch.get_num_threads()}


def resolve_config(config: dict) -> dict:
    cfg = {**DEFAULTS, **config}
    for name in ('seed', 'steps', 'batch_size', 'batches_per_scene', 'eval_every', 'checkpoint_every', 'threads'):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < (0 if name == 'seed' else 1):
            raise ValueError(f'invalid {name}')
    for name in ('learning_rate', 'weight_decay', 'huber_weight'):
        if not np.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f'invalid {name}')
    for name in ('dataset_root', 'schema_path', 'splits_path', 'samples_path', 'init_checkpoint'):
        if cfg.get(name):
            cfg[name] = str(Path(cfg[name]).resolve())
    if cfg.get('splits_path'):
        splits = json.loads(Path(cfg['splits_path']).read_text())
        if 'train_catalog_ids' not in cfg:
            cfg['train_catalog_ids'] = splits['train_subsets'][str(cfg.get('subset', 40))]
        if 'validation_catalog_ids' not in cfg:
            cfg['validation_catalog_ids'] = splits['pilot_validation']
    if not cfg.get('train_catalog_ids'):
        raise ValueError('explicit train_catalog_ids or frozen splits_path required')
    return cfg


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Natural-distribution AP (step-integral PR-AUC), with threshold ties grouped."""
    y, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all() or not len(y):
        raise ValueError('finite nonempty ranking inputs required')
    order = np.argsort(-scores, kind='stable')
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    positives = int(y.sum())
    tp = np.cumsum(ys)[ends].astype(float)
    precision = tp / (ends + 1)
    recall = tp / positives if positives else np.zeros(len(ends))
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else 0.
    k = max(1, positives)
    gain = 1. / np.log2(np.arange(k) + 2)
    return {'rows': len(y), 'positives': positives, 'average_precision': ap,
            'pr_auc_definition': 'average precision, grouped score thresholds',
            'precision_at_positive_count': float(ys[:k].mean()),
            'recall_at_positive_count': float(ys[:k].sum() / positives) if positives else 0.,
            'ndcg_at_positive_count': float(np.dot(ys[:k], gain) / gain.sum()) if positives else 0.}


@torch.no_grad()
def evaluate_static(model, reader, records, normalizer, *, device='cpu', batch_size=8192):
    model.eval()
    rows = []
    for record in records:
        if record['split'] != 'validation':
            raise ValueError('checkpoint selection may only read validation')
        arrays = reader._load_actions(record)
        scores = []
        for offset in range(0, len(arrays['features']), batch_size):
            x = transform(arrays['features'][offset:offset + batch_size], normalizer)
            scores.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
        metrics = ranking_metrics(arrays['grid_selected'], np.concatenate(scores))
        rows.append({'scene_id': record['scene_id'], 'catalog_id': record['catalog_id'],
                     'group': 'stress' if record.get('stress_type') or record.get('distribution') == 'stress' else 'core', **metrics})
    metric_names = ('average_precision', 'precision_at_positive_count', 'recall_at_positive_count', 'ndcg_at_positive_count')
    groups = {}
    for group in ('all', 'core', 'stress'):
        members = [r for r in rows if group == 'all' or r['group'] == group]
        groups[group] = {'scenes': len(members), **{k: float(np.mean([r[k] for r in members])) if members else None for k in metric_names}}
    model.train()
    return {'scene_metrics': rows, 'scene_equal_means': groups,
            'plan_metrics': None, 'plan_metrics_status': 'requires separate full planner and independent validator'}


class DynamicCorpus:
    """Small bounded first-stage counterfactual corpus; preserve all failures in counts."""
    def __init__(self, path, reader, train_catalog_ids, *, schema):
        records = select_records(reader, train_catalog_ids, 'train')
        allowed = {(r['catalog_id'], r['scene_id']) for r in records}
        self.groups, self.counts = [], Counter()
        grouped = defaultdict(list)
        contexts, versions, seen_candidates = {}, set(), set()
        with Path(path).open() as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                self.counts['all_rows'] += 1
                if r.get('split') != 'train' or (r.get('catalog_id'), r.get('scene_id')) not in allowed:
                    raise ValueError(f'dynamic split/catalog leakage on line {lineno}')
                key = (r['catalog_id'], r['scene_id'], r['parent_state_hash'])
                if (r.get('static_feature_names') != schema['static_names']
                        or r.get('dynamic_feature_names') != schema['dynamic_names']):
                    raise ValueError('counterfactual feature column order differs from schema')
                if not isinstance(r.get('operation'), dict):
                    raise ValueError('counterfactual row needs operation parameters')
                candidate_key = (*key, digest(r['operation']))
                if candidate_key in seen_candidates:
                    raise ValueError('duplicate same-state counterfactual operation')
                seen_candidates.add(candidate_key)
                context = {k: r.get(k) for k in ('reference_state_hash', 'repair_budget', 'pool_sha256', 'price_sha256')}
                if key in contexts and contexts[key] != context:
                    raise ValueError('same-state labels have different repair/pool/price/reference context')
                contexts[key] = context
                if r.get('status') != 'completed' or r.get('censored', False) or r.get('delta_Q') is None:
                    self.counts['excluded_' + str(r.get('status', 'missing_status'))] += 1
                    continue
                if not np.isfinite(r['delta_Q']):
                    raise ValueError('nonfinite dynamic label')
                versions.add(tuple(r.get(k) for k in ('feature_version', 'pool_version', 'price_version')))
                x, d = np.asarray(r['static_features']), np.asarray(r['dynamic_features'])
                if x.shape != (20,) or d.ndim != 1 or not len(d) or not np.isfinite(x).all() or not np.isfinite(d).all():
                    raise ValueError('invalid dynamic input features')
                grouped[key].append(r)
                self.counts['completed_rows'] += 1
        if len(versions) != 1 or any(v is None for v in next(iter(versions), ())):
            raise ValueError('counterfactual feature/pool/price versions must be present and identical')
        self.versions = dict(zip(('feature_version', 'pool_version', 'price_version'), next(iter(versions))))
        dimensions = {len(r['dynamic_features']) for rows in grouped.values() for r in rows}
        if len(dimensions) != 1:
            raise ValueError('inconsistent dynamic dimensions')
        self.dynamic_dim = dimensions.pop()
        for key, rows in sorted(grouped.items()):
            q = np.asarray([r['delta_Q'] for r in rows], np.float32)
            ii, jj = np.triu_indices(len(q), 1)
            keep = q[ii] != q[jj]
            ii, jj = ii[keep], jj[keep]
            if not len(ii):
                self.counts['unrankable_states'] += 1
                continue
            good = q[ii] > q[jj]
            pairs = np.stack((np.where(good, ii, jj), np.where(good, jj, ii)), axis=1)
            self.groups.append({'key': key, 'static': np.asarray([r['static_features'] for r in rows], np.float32),
                                'dynamic': np.asarray([r['dynamic_features'] for r in rows], np.float32),
                                'labels': q, 'pairs': pairs})
        if not self.groups:
            raise ValueError('no completed same-state unequal-gain pairs; do not train on censored zeros')
        self.by_catalog = defaultdict(list)
        for i, g in enumerate(self.groups):
            self.by_catalog[g['key'][0]].append(i)
        self.catalogs = sorted(self.by_catalog)
        self.counts['rankable_states'] = len(self.groups)
        self.counts['pairs'] = sum(len(g['pairs']) for g in self.groups)
        self.sha256 = file_sha256(path)

    def group(self, step, seed):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 9503]))
        cat = self.catalogs[int(rng.integers(len(self.catalogs)))]
        return self.groups[int(rng.choice(self.by_catalog[cat]))]

    def normalizer(self, key):
        # Equal catalog, then equal retained state, then uniform completed candidate.
        means, variances = [], []
        for catalog in self.catalogs:
            group_means, group_vars = [], []
            for i in self.by_catalog[catalog]:
                x = self.groups[i][key].astype(np.float64)
                group_means.append(x.mean(0)); group_vars.append(x.var(0))
            m = np.mean(group_means, axis=0)
            v = np.mean([v + (gm - m) ** 2 for gm, v in zip(group_means, group_vars)], axis=0)
            means.append(m); variances.append(v)
        mean = np.mean(means, axis=0)
        var = np.mean([v + (m - mean) ** 2 for m, v in zip(means, variances)], axis=0)
        scale = np.sqrt(np.maximum(var, 0)); scale[scale < 1e-12] = 1
        return {'version': 1, 'source_split': 'train', 'mean': mean.tolist(), 'scale': scale.tolist(),
                'catalog_ids': self.catalogs, 'samples_sha256': self.sha256,
                'weighting': 'equal_catalog_then_equal_state_then_uniform_completed_candidate',
                'feature_names': list(FEATURES) if key == 'static' else None}


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def _rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def run_training(config: dict, output: str | Path, *, stage='static', resume=None,
                 stop_after: int | None = None, max_wall_s: float | None = None) -> dict:
    """Run until frozen steps, bounded wall time, or explicit smoke stop; resume exact next step.

    output may only already exist with --resume. Each attempt/checkpoint is kept.
    max_wall_s is a cooperative boundary between steps, not a subprocess hard kill.
    """
    cfg = resolve_config(config)
    if stage not in ('static', 'dynamic'):
        raise ValueError('stage must be static or dynamic')
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop_after must be positive')
    if max_wall_s is not None and max_wall_s <= 0:
        raise ValueError('max_wall_s must be positive')
    out = Path(output).resolve()
    if out.exists() and (resume is None or not (out / 'config.json').is_file()):
        raise FileExistsError('output exists; resume a matching run or use a new directory')
    out.mkdir(parents=True, exist_ok=resume is not None)
    attempt = uuid.uuid4().hex
    start, cpu_start = time.perf_counter(), time.process_time()
    step, saved_path = 0, None
    append_json(out / 'attempts.jsonl', {'attempt': attempt, 'status': 'started', 'stage': stage,
                                       'resume': str(resume) if resume else None, 'unix_time': time.time()})
    try:
        torch.set_num_threads(cfg['threads'])
        torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
        torch.use_deterministic_algorithms(True)
        device = cfg['device']
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        reader = DatasetReader(cfg['dataset_root'])
        train = select_records(reader, cfg['train_catalog_ids'], 'train')
        schema = json.loads(Path(cfg['schema_path']).read_text())
        if schema.get('static_names') != list(FEATURES):
            raise ValueError('feature schema must preserve canonical static column order')
        source_names = ('training.py', 'static_data.py', 'model.py', 'inference.py')
        identity = {'stage': stage, 'config_sha256': digest(cfg), 'schema_sha256': digest(schema),
                    'manifest_sha256': file_sha256(Path(cfg['dataset_root']) / 'manifest.json'),
                    'train_records_sha256': digest(train),
                    'source_sha256': {n: file_sha256(Path(__file__).with_name(n)) for n in source_names},
                    'torch_version': str(torch.__version__), 'numpy_version': str(np.__version__)}
        if cfg.get('splits_path'):
            identity['splits_sha256'] = file_sha256(cfg['splits_path'])
        restored = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
        corpus, val, dynamic_norm = None, None, None
        if stage == 'dynamic':
            corpus = DynamicCorpus(cfg['samples_path'], reader, cfg['train_catalog_ids'], schema=schema)
            if len(schema.get('dynamic_names', [])) != corpus.dynamic_dim:
                raise ValueError('dynamic schema dimension differs from samples')
            if schema.get('version') and schema['version'] != corpus.versions['feature_version']:
                raise ValueError('dynamic feature schema version differs from samples')
            identity['samples_sha256'] = corpus.sha256
        if cfg.get('init_checkpoint'):
            identity['initialization_sha256'] = file_sha256(cfg['init_checkpoint'])
        if restored and restored['identity'] != identity:
            raise ValueError('checkpoint identity/config/data/schema/source differs; resume refused')
        if restored:
            normalizer, dynamic_norm = restored['normalizer'], restored['dynamic_normalizer']
        elif stage == 'static':
            normalizer = fit_scene_normalizer(reader, train)
        else:
            normalizer, dynamic_norm = corpus.normalizer('static'), corpus.normalizer('dynamic')
        init = None
        if stage == 'dynamic' and cfg.get('init_checkpoint'):
            init = torch.load(cfg['init_checkpoint'], map_location='cpu', weights_only=True)
            if init['stage'] != 'static' or init['normalizer'].get('feature_names') != list(FEATURES):
                raise ValueError('initialization requires canonical static encoder checkpoint')
            if not set(init['normalizer']['catalog_ids']).issubset(set(cfg['train_catalog_ids'])):
                raise ValueError('pretraining checkpoint exceeds declared train subset')
            normalizer = init['normalizer']  # preserve encoder input coordinate system
        if stage == 'static':
            val = select_records(reader, cfg['validation_catalog_ids'], 'validation')
            stream = BalancedStream(reader, train, normalizer, seed=cfg['seed'], batch_size=cfg['batch_size'],
                                    batches_per_scene=cfg['batches_per_scene'])
        model = CandidateMLP(corpus.dynamic_dim if corpus else 0).to(device)
        if init and not restored:
            model.encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in init['model_state'].items() if k.startswith('encoder.')})
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        best_ap, best_checkpoint = -1., None
        elapsed_previous, cpu_previous = 0., 0.
        if restored:
            model.load_state_dict(restored['model_state']); optimizer.load_state_dict(restored['optimizer_state'])
            step = restored['step']; best_ap = restored['best_validation_ap']; best_checkpoint = restored.get('best_checkpoint')
            elapsed_previous, cpu_previous = restored['elapsed_s'], restored['cpu_s']
            _restore_rng(restored['rng'])
        else:
            write_json(out / 'config.json', cfg)
            write_json(out / 'feature_schema.json', schema)
            write_json(out / 'normalizer.json', normalizer)
            if dynamic_norm is not None:
                write_json(out / 'dynamic_normalizer.json', dynamic_norm)
            write_json(out / 'identity.json', identity)
        write_json(out / f'environment-{attempt}.json', environment())
        steps_started = step
        last_validation, loss_value = None, None

        def save_checkpoint(status):
            nonlocal saved_path, best_checkpoint
            _sync(device)
            saved_path = out / f'checkpoint-step-{step:06d}-{attempt}.pt'
            if last_validation is not None and last_validation['selected']:
                best_checkpoint = saved_path.name
            payload = {'format_version': 1, 'stage': stage, 'step': step, 'next_batch_step': step,
                       'identity': identity, 'config': cfg, 'schema': schema, 'normalizer': normalizer,
                       'dynamic_normalizer': dynamic_norm, 'dynamic_dim': model.dynamic_dim,
                       'feature_versions': corpus.versions if corpus else {'pool_version': 'synthetic-v0.2-max12'},
                       'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'rng': _rng_state(),
                       'best_validation_ap': best_ap, 'best_checkpoint': best_checkpoint,
                       'elapsed_s': elapsed_previous + time.perf_counter() - start,
                       'cpu_s': cpu_previous + time.process_time() - cpu_start,
                       'status': status, 'loss': loss_value}
            # Never overwrite a prior checkpoint, including an interrupted attempt.
            with saved_path.open('xb') as fh:
                torch.save(payload, fh)
            sha = file_sha256(saved_path)
            write_json(saved_path.with_suffix('.sha256.json'), {'file': saved_path.name, 'sha256': sha})
            pointer = {'checkpoint': saved_path.name, 'sha256': sha, 'step': step, 'status': status,
                       'best_checkpoint': best_checkpoint}
            tmp = out / f'.latest-{attempt}.json'
            write_json(tmp, pointer, exclusive=False); os.replace(tmp, out / 'latest.json')
            return pointer

        last_saved_step = None
        while step < cfg['steps']:
            if stop_after is not None and step - steps_started >= stop_after:
                break
            if max_wall_s is not None and time.perf_counter() - start >= max_wall_s:
                break
            model.train(); optimizer.zero_grad(set_to_none=True)
            if stage == 'static':
                batch = stream.batch(step)
                x = torch.from_numpy(batch['features']).to(device)
                y = torch.from_numpy(batch['labels']).to(device)
                loss = F.binary_cross_entropy_with_logits(model(x), y,
                          pos_weight=torch.tensor(normalizer['pos_weight'], device=device))
                data_cursor = {'step': step, 'scene_id': batch['scene_id']}
            else:
                group = corpus.group(step, cfg['seed'])
                x = torch.from_numpy(transform(group['static'], normalizer)).to(device)
                d = torch.from_numpy(transform(group['dynamic'], dynamic_norm, dim=corpus.dynamic_dim)).to(device)
                scores = model(x, d)
                pairs = torch.from_numpy(group['pairs']).to(device)
                loss = F.softplus(-(scores[pairs[:, 0]] - scores[pairs[:, 1]])).mean()
                if cfg['huber_weight']:
                    labels = torch.from_numpy(group['labels']).to(device)
                    loss = loss + cfg['huber_weight'] * F.smooth_l1_loss(scores, labels)
                data_cursor = {'step': step, 'parent_state': list(group['key'])}
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward(); optimizer.step(); step += 1
            loss_value = float(loss.detach().cpu())
            append_json(out / 'train.jsonl', {'attempt': attempt, 'step': step, 'loss': loss_value,
                                             'data_cursor': data_cursor, 'wall_s': time.perf_counter() - start})
            last_validation = None
            if stage == 'static' and (step % cfg['eval_every'] == 0 or step == cfg['steps']):
                metrics = evaluate_static(model, reader, val, normalizer, device=device, batch_size=cfg['batch_size'])
                ap = metrics['scene_equal_means']['all']['average_precision']
                last_validation = {'attempt': attempt, 'step': step, 'selected': ap > best_ap, **metrics}
                best_ap = max(best_ap, ap)
                append_json(out / 'validation.jsonl', last_validation)
            if step % cfg['checkpoint_every'] == 0 or last_validation is not None:
                save_checkpoint('running'); last_saved_step = step
        status = 'completed_frozen_steps' if step == cfg['steps'] else 'stopped_resumable'
        if last_saved_step != step:
            result = save_checkpoint(status)
        else:
            result = json.loads((out / 'latest.json').read_text())
            result['status'] = status
            tmp = out / f'.latest-{attempt}.json'; write_json(tmp, result); os.replace(tmp, out / 'latest.json')
        result.update({'attempt': attempt, 'stage': stage, 'elapsed_attempt_s': time.perf_counter() - start,
                       'cpu_attempt_s': time.process_time() - cpu_start,
                       'parameter_count': sum(p.numel() for p in model.parameters()),
                       'dynamic_counts': dict(corpus.counts) if corpus else None,
                       'elapsed_total_s': elapsed_previous + time.perf_counter() - start,
                       'cpu_hours_total': (cpu_previous + time.process_time() - cpu_start) / 3600.,
                       'gpu_reserved_hours_total': (elapsed_previous + time.perf_counter() - start) / 3600.
                           if str(device).startswith('cuda') else 0.,
                       'checkpoint_bytes': saved_path.stat().st_size})
        write_json(out / f'result-{attempt}.json', result)
        append_json(out / 'attempts.jsonl', {**result, 'unix_time': time.time()})
        card = (f'# Neural local-search model artifact\n\nStage: {stage}; seed: {cfg["seed"]}; completed gradient steps: {step}/{cfg["steps"]}.\n\n'
                f'Checkpoint: `{result["checkpoint"]}`; SHA-256: `{result["sha256"]}`.\n\n'
                f'Parameters: {result["parameter_count"]}; device: {device}; attempt wall seconds: {result["elapsed_attempt_s"]:.3f}; '
                f'attempt CPU seconds: {result["cpu_attempt_s"]:.3f}; total CPU hours: {result["cpu_hours_total"]:.6f}; '
                f'total GPU reserved hours: {result["gpu_reserved_hours_total"]:.6f}; checkpoint bytes: {result["checkpoint_bytes"]}. CUDA elapsed is reserved-device wall time, not utilization.\n\n'
                'Static labels indicate teacher selection, never legality. Weighted-BCE scores are not calibrated probabilities. '
                'Static features were trained on the max-12-window teacher pool; dynamic inputs are recomputed with their recorded pool/price version.\n\n'
                'Dynamic supervision, when present, compares completed candidates from the same state and repair budget; '
                'censored evaluations are retained in the sample file and excluded from pairs.\n\n'
                'Training throughput and surrogate validation metrics do not establish schedule quality. '
                'No formal test acceptance, independent Skyfield validation, or replacement of H+ is claimed. '
                'Dynamic checkpoint selection requires the frozen validation planner evaluation.\n')
        (out / f'MODEL_CARD-{attempt}.md').write_text(card, encoding='utf-8')
        hashes = {p.name: file_sha256(p) for p in out.iterdir()
                  if p.is_file() and p.suffix in ('.pt', '.json', '.jsonl', '.md') and not p.name.startswith('artifacts-')}
        write_json(out / f'artifacts-{attempt}.json', hashes)
        return result
    except BaseException as exc:
        append_json(out / 'attempts.jsonl', {'attempt': attempt, 'status': 'failed' if not isinstance(exc, KeyboardInterrupt) else 'interrupted',
                    'step': step, 'error_type': type(exc).__name__, 'error': str(exc),
                    'latest_saved_checkpoint': str(saved_path) if saved_path else None,
                    'elapsed_s': time.perf_counter() - start, 'unix_time': time.time()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('static', 'dynamic'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='bounded smoke interruption; frozen total steps unchanged')
    parser.add_argument('--max-wall-s', type=float, help='cooperative stop between steps, checkpoint retained')
    args = parser.parse_args(argv)
    result = run_training(json.loads(args.config.read_text()), args.output, stage=args.stage,
                          resume=args.resume, stop_after=args.stop_after, max_wall_s=args.max_wall_s)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
