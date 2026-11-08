"""Guard leakage, complete candidate mapping, exact resume, and censored labels."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from obsplan.famou_h_eval import FEATURES
from obsplan.training_data.loader import DatasetReader
from obsplan.neural_lns.static_data import BalancedStream, fit_scene_normalizer, select_records
from obsplan.neural_lns.training import DynamicCorpus, ranking_metrics, run_training
from obsplan.neural_lns.inference import Ranker, main as predict_main
from test_training_loader import write_corpus


def setup(tmp_path, steps=4):
    data = tmp_path / 'data'; data.mkdir()
    write_corpus(data)
    schema = tmp_path / 'schema.json'
    schema.write_text(json.dumps({'static_names': FEATURES, 'dynamic_names': ['gap', 'mask']}))
    config = {'dataset_root': str(data), 'schema_path': str(schema),
              'train_catalog_ids': ['cat_a', 'cat_b'], 'validation_catalog_ids': ['cat_c'],
              'steps': steps, 'batch_size': 8, 'batches_per_scene': 1, 'eval_every': 2,
              'checkpoint_every': 2, 'seed': 2026, 'device': 'cpu', 'threads': 1}
    return data, config


def checkpoint(out, result):
    return out / result['checkpoint']


def test_subset_scene_weighting_and_stream_mapping(tmp_path):
    root, _ = setup(tmp_path)
    reader = DatasetReader(root)
    records = select_records(reader, ['cat_a', 'cat_b'], 'train')
    norm = fit_scene_normalizer(reader, records)
    assert norm['mean'][0] == 2.0  # (scene mean 1 + scene mean 3)/2, not row-weighted 1.75
    assert norm['mean'][-1] == 7 and norm['scale'][-1] == 1
    subset = fit_scene_normalizer(reader, select_records(reader, ['cat_a'], 'train'))
    assert subset['mean'][0] == 1 and subset['catalog_ids'] == ['cat_a']
    with pytest.raises(ValueError, match='outside required'):
        select_records(reader, ['cat_c'], 'train')
    with pytest.raises(ValueError, match='train records'):
        fit_scene_normalizer(reader, reader.records('validation'))
    stream = BalancedStream(reader, records, norm, batch_size=8, batches_per_scene=2)
    batches = [stream.batch(i) for i in range(4)]
    assert sorted([b['scene_id'] for b in batches]) == ['a', 'a', 'b', 'b']
    for b in batches:
        assert np.array_equal(b['labels'].astype(bool), b['candidate_id'] == 0)
        np.testing.assert_array_equal(stream.batch(b['step'])['candidate_id'], b['candidate_id'])


def test_resume_matches_uninterrupted_optimizer_rng_and_predictions(tmp_path):
    _, cfg = setup(tmp_path)
    full, split = tmp_path / 'full', tmp_path / 'split'
    end_full = run_training(cfg, full)
    partial = run_training(cfg, split, stop_after=2)
    assert partial['status'] == 'stopped_resumable'
    partial_path = checkpoint(split, partial)
    continued = run_training(cfg, split, resume=partial_path)
    a = torch.load(checkpoint(full, end_full), weights_only=True)
    b = torch.load(checkpoint(split, continued), weights_only=True)
    assert a['step'] == b['step'] == 4
    for key in a['model_state']:
        torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
    assert partial_path.is_file()  # attempt/checkpoint retained
    ranker = Ranker(checkpoint(split, continued))
    test_x = np.ones((3, 20), np.float32)
    from obsplan.neural_lns.static_data import transform
    expected = ranker.model(torch.from_numpy(transform(test_x, ranker.normalizer))).detach().numpy()
    np.testing.assert_array_equal(ranker.score(test_x), expected)
    with pytest.raises(ValueError, match='identity/config'):
        run_training({**cfg, 'seed': 2027}, split, resume=checkpoint(split, continued))
    failures = [json.loads(x) for x in (split / 'attempts.jsonl').read_text().splitlines()]
    assert failures[-1]['status'] == 'failed'
    with pytest.raises(FileExistsError):
        run_training(cfg, full)
    with pytest.raises(ValueError, match='SHA-256'):
        Ranker(checkpoint(full, end_full), expected_sha256='0' * 64)


def dynamic_rows():
    rows = []
    for cat, scene in [('cat_a', 'a'), ('cat_b', 'b')]:
        for i, delta in enumerate([0.1, -0.1, None]):
            rows.append({'catalog_id': cat, 'scene_id': scene, 'split': 'train',
                         'parent_state_hash': 'state', 'reference_state_hash': 'reference',
                         'repair_budget': {'max_trials': 64, 'timeout_s': 2},
                         'pool_sha256': 'pool', 'price_sha256': 'prices',
                         'feature_version': 'neural-lns-features-v1', 'pool_version': 'full-windows-v1',
                         'price_version': 'hplus-fixed-mu-v1', 'status': 'completed' if delta is not None else 'timeout',
                         'censored': delta is None, 'delta_Q': delta,
                         'static_feature_names': FEATURES, 'dynamic_feature_names': ['gap', 'mask'],
                         'operation': {'index': i},
                         'static_features': [float(i)] * 20, 'dynamic_features': [float(i), 1.]})
    return rows


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_dynamic_censoring_context_and_split_are_enforced(tmp_path):
    root, cfg = setup(tmp_path)
    path = tmp_path / 'samples.jsonl'; rows = dynamic_rows(); save_rows(path, rows)
    corpus = DynamicCorpus(path, DatasetReader(root), cfg['train_catalog_ids'], schema=json.loads(Path(cfg['schema_path']).read_text()))
    assert corpus.counts['excluded_timeout'] == 2
    assert corpus.counts['pairs'] == 2
    assert len(corpus.groups) == 2
    rows[1]['repair_budget']['timeout_s'] = 3
    save_rows(path, rows)
    with pytest.raises(ValueError, match='different repair'):
        DynamicCorpus(path, DatasetReader(root), cfg['train_catalog_ids'], schema=json.loads(Path(cfg['schema_path']).read_text()))
    rows = dynamic_rows(); rows.append(dict(rows[0])); save_rows(path, rows)
    with pytest.raises(ValueError, match='duplicate same-state'):
        DynamicCorpus(path, DatasetReader(root), cfg['train_catalog_ids'], schema=json.loads(Path(cfg['schema_path']).read_text()))
    rows = dynamic_rows(); rows[0]['dynamic_feature_names'] = ['mask', 'gap']; save_rows(path, rows)
    with pytest.raises(ValueError, match='column order'):
        DynamicCorpus(path, DatasetReader(root), cfg['train_catalog_ids'], schema=json.loads(Path(cfg['schema_path']).read_text()))
    rows = dynamic_rows(); rows[0]['split'] = 'test'; save_rows(path, rows)
    with pytest.raises(ValueError, match='leakage'):
        DynamicCorpus(path, DatasetReader(root), cfg['train_catalog_ids'], schema=json.loads(Path(cfg['schema_path']).read_text()))


def test_dynamic_transfer_random_resume_and_prediction_cli(tmp_path, capsys):
    _, cfg = setup(tmp_path)
    static_out = tmp_path / 'static'; s = run_training(cfg, static_out)
    samples = tmp_path / 'samples.jsonl'; save_rows(samples, dynamic_rows())
    dc = {**cfg, 'samples_path': str(samples), 'init_checkpoint': str(checkpoint(static_out, s))}
    out = tmp_path / 'dynamic'; first = run_training(dc, out, stage='dynamic', stop_after=2)
    last = run_training(dc, out, stage='dynamic', resume=checkpoint(out, first))
    cp = checkpoint(out, last)
    ranker = Ranker(cp, expected_versions={'pool_version': 'full-windows-v1'})
    assert ranker.model.dynamic_dim == 2
    assert ranker.normalizer == Ranker(checkpoint(static_out, s)).normalizer
    with pytest.raises(ValueError, match='versions differ'):
        Ranker(cp, expected_versions={'pool_version': 'teacher-pool'})
    inp, pred = tmp_path / 'input.npz', tmp_path / 'pred.npz'
    np.savez(inp, features=np.ones((3, 20), np.float32), dynamic_features=np.ones((3, 2), np.float32),
             candidate_id=np.array([7, 3, 12]))
    predict_main(['--checkpoint', str(cp), '--input', str(inp), '--output', str(pred), '--batch-size', '2'])
    report = json.loads(capsys.readouterr().out)
    assert report['rows'] == 3 and report['inference_s'] > 0
    with np.load(pred) as z:
        np.testing.assert_array_equal(z['candidate_id'], [7, 3, 12])
        assert z['scores'].shape == (3,)
    random_out = tmp_path / 'random'
    random_result = run_training({**cfg, 'samples_path': str(samples)}, random_out, stage='dynamic')
    assert random_result['dynamic_counts']['pairs'] == 2


def test_ranking_metrics_ties_and_natural_distribution():
    m = ranking_metrics(np.array([1, 0, 0, 0]), np.array([0., 0., 0., 0.]))
    assert m['average_precision'] == .25
    assert ranking_metrics(np.array([1, 0, 1]), np.array([3., 1., 2.]))['average_precision'] == 1.
