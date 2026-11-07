"""Protect scene boundaries, train-only statistics and label interpretation."""
import json

import numpy as np
import pytest

from obsplan.famou_h_eval import FEATURES
from obsplan.training_data.loader import DatasetReader, Normalizer, main


def write_corpus(tmp_path, specs=(('train', 'a', 5, 1.), ('train', 'b', 3, 3.),
                                   ('validation', 'c', 4, 1000.), ('test', 'd', 2, -2000.))):
    scenes = []
    for split, name, rows, value in specs:
        record = {"scene_id": name, "catalog_id": f"cat_{name}", "split": split,
                  "path": f"scenes/{name}"}
        path = tmp_path / record["path"]
        path.mkdir(parents=True)
        selected = np.zeros(rows, bool)
        selected[:1] = True
        features = np.full((rows, 20), value, np.float32)
        features[:, -1] = 7  # constant feature: no NaNs after normalization
        arrays = {
            "features": features,
            "candidate_window": np.arange(rows, dtype=np.int32),
            "candidate_duration_cells": np.full(rows, 4, np.int16),
            "rule_score_uncovered": np.ones(rows, np.float32),
            "rule_score_covered": np.ones(rows, np.float32),
            "grid_selected": selected,
            "grid_selection_rank": np.where(selected, 0, -1).astype(np.int32),
            "grid_start_s": np.where(selected, 30, -1).astype(np.int32),
            "grid_duration_s": np.where(selected, 120, -1).astype(np.int32),
        }
        np.savez_compressed(path / "actions.npz", **arrays)
        scenes.append(record)
    (tmp_path / "manifest.json").write_text(json.dumps({"feature_names": FEATURES, "scenes": scenes}))
    return scenes


def amend_manifest(root, mutate):
    path = root / "manifest.json"
    obj = json.loads(path.read_text())
    mutate(obj)
    path.write_text(json.dumps(obj))


def amend_actions(root, name, mutate):
    path = root / "scenes" / name / "actions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def test_train_only_streaming_stats_and_roundtrip(tmp_path):
    write_corpus(tmp_path)
    reader = DatasetReader(tmp_path)
    stats = reader.fit_normalizer(batch_size=2)
    assert stats.row_count == 8
    assert stats.scene_ids == ('a', 'b')
    # Holdout values deliberately dominate if train-only slicing is broken.
    expected = np.array([1.] * 5 + [3.] * 3)
    np.testing.assert_allclose(stats.mean[:-1], expected.mean())
    np.testing.assert_allclose(stats.scale[:-1], expected.std())
    assert stats.mean[-1] == 7 and stats.scale[-1] == 1
    train = np.concatenate([b["features"] for b in reader.iter_batches("train", 2, normalizer=stats)])
    np.testing.assert_allclose(train.mean(0), 0, atol=1e-7)
    assert np.isfinite(train).all() and train.dtype == np.float32
    path = tmp_path / "normalizer.json"
    stats.save(path)
    restored = Normalizer.load(path)
    np.testing.assert_array_equal(restored.scale, stats.scale)
    with pytest.raises(FileExistsError):
        stats.save(path)


def test_batches_keep_scene_boundaries_and_remainders(tmp_path):
    write_corpus(tmp_path)
    reader = DatasetReader(tmp_path)
    batches = list(reader.iter_batches("train", batch_size=3))
    assert [(b['scene_id'], len(b['features']), b['row_offset']) for b in batches] == [
        ('a', 3, 0), ('a', 2, 3), ('b', 3, 0)]
    assert len(list(reader.iter_batches("train", 2, max_batches=1))) == 1
    assert all(b['split'] == 'train' for b in batches)


@pytest.mark.parametrize("method,args", [('records', ('valid',)),
                                        ('iter_batches', ('bogus',)),
                                        ('iter_batches', ('train', 0)),
                                        ('iter_batches', ('train', -1))])
def test_bad_split_or_batch_rejected(tmp_path, method, args):
    write_corpus(tmp_path)
    reader = DatasetReader(tmp_path)
    with pytest.raises(ValueError):
        result = getattr(reader, method)(*args)
        if method == 'iter_batches':
            list(result)


def test_empty_train_cannot_fit(tmp_path):
    write_corpus(tmp_path, [('train', 'a', 0, 1.), ('val', 'b', 1, 2.)])
    with pytest.raises(ValueError, match='no candidate rows'):
        DatasetReader(tmp_path).fit_normalizer()


@pytest.mark.parametrize("change,match", [
    (lambda m: m['scenes'][2].update(catalog_id='cat_a'), 'catalog leakage'),
    (lambda m: m['scenes'][2].update(scene_id='a'), 'duplicate scene'),
    (lambda m: m['scenes'][2].update(path='scenes/a'), 'duplicate scene'),
    (lambda m: m['scenes'][2].update(path='../elsewhere'), 'inside dataset'),
    (lambda m: m['scenes'][2].update(path='/tmp/elsewhere'), 'inside dataset'),
    (lambda m: m.update(feature_names=list(reversed(FEATURES))), 'canonical'),
])
def test_manifest_corruption_or_leakage_rejected(tmp_path, change, match):
    write_corpus(tmp_path)
    amend_manifest(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        DatasetReader(tmp_path)


@pytest.mark.parametrize("change,match", [
    (lambda a: a['features'].__setitem__((0, 0), np.nan), 'non-finite'),
    (lambda a: a['rule_score_covered'].__setitem__(0, np.inf), 'non-finite'),
    (lambda a: a.update(features=a['features'].astype(np.float64)), 'dtype'),
    (lambda a: a.update(candidate_window=a['candidate_window'][:-1]), 'shape'),
    (lambda a: a['grid_selection_rank'].__setitem__(0, -2), 'teacher labels'),
    (lambda a: a['grid_start_s'].__setitem__(1, 0), 'teacher labels'),
    (lambda a: a.pop('grid_selected'), 'missing action fields'),
])
def test_invalid_features_and_labels_fail_before_training(tmp_path, change, match):
    write_corpus(tmp_path)
    amend_actions(tmp_path, 'a', change)
    with pytest.raises(ValueError, match=match):
        list(DatasetReader(tmp_path).iter_batches('train'))


def test_normalizer_rejects_validation_statistics(tmp_path):
    write_corpus(tmp_path)
    stats = DatasetReader(tmp_path).fit_normalizer()
    obj = stats.as_dict()
    obj['source_split'] = 'val'
    path = tmp_path / 'bad_stats.json'
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='fitted on train'):
        Normalizer.load(path)


def test_normalizer_manifest_mismatch(tmp_path):
    write_corpus(tmp_path)
    stats = DatasetReader(tmp_path).fit_normalizer()
    amend_manifest(tmp_path, lambda m: m['scenes'][0].update(scene_id='different'))
    with pytest.raises(ValueError, match='different train manifest'):
        list(DatasetReader(tmp_path).iter_batches('test', normalizer=stats))


def test_smoke_command_reads_all_splits_without_writing(tmp_path, capsys):
    write_corpus(tmp_path)
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}
    main([str(tmp_path), '--smoke', '--batch-size', '2'])
    output = json.loads(capsys.readouterr().out)
    assert output['mode'] == 'first_batch_per_split'
    assert output['normalizer']['row_count'] == 8
    assert set(output['splits']) == {'train', 'validation', 'test'}
    assert all(s['read_rows'] == 2 for s in output['splits'].values())
    assert 'not a legality' in output['label_semantics']
    assert before == {p.relative_to(tmp_path) for p in tmp_path.rglob('*')}


def test_validation_alias_is_normalized(tmp_path):
    write_corpus(tmp_path, [('train', 'a', 2, 1.), ('val', 'b', 2, 2.)])
    reader = DatasetReader(tmp_path)
    assert reader.records('val') == reader.records('validation')
    batch = next(reader.iter_batches('val'))
    assert batch['split'] == 'validation'


def test_repeated_candidate_preserves_first_rank_gaps_and_task_mapping(tmp_path):
    write_corpus(tmp_path, [('train', 'a', 3, 1.)])
    def change(a):
        a['grid_selected'][:] = [True, True, False]
        a['grid_selection_rank'][:] = [0, 2, -1]
        a['grid_start_s'][:] = [30, 60, -1]
        a['grid_duration_s'][:] = [120, 120, -1]
        a['grid_selected_count'] = np.array([2, 1, 0], np.int32)
        a['grid_task_candidate'] = np.array([0, 0, 1], np.int32)
    amend_actions(tmp_path, 'a', change)
    reader = DatasetReader(tmp_path)
    _, arrays = next(reader.iter_scenes('train'))
    np.testing.assert_array_equal(arrays['grid_task_candidate'], [0, 0, 1])
    batch = next(reader.iter_batches('train', 2))
    assert 'grid_task_candidate' not in batch  # task rows are not candidate rows
    np.testing.assert_array_equal(batch['grid_selected_count'], [2, 1])
    np.testing.assert_array_equal(batch['grid_selection_rank'], [0, 2])


def test_inconsistent_task_mapping_rejected(tmp_path):
    write_corpus(tmp_path, [('train', 'a', 3, 1.)])
    amend_actions(tmp_path, 'a', lambda a: a.update(grid_task_candidate=np.array([1], np.int32)))
    with pytest.raises(ValueError, match='task mapping disagrees'):
        list(DatasetReader(tmp_path).iter_scenes('train'))


def test_unique_selected_ranks_required_without_task_mapping(tmp_path):
    write_corpus(tmp_path, [('train', 'a', 3, 1.)])
    def change(a):
        a['grid_selected'][:2] = True
        a['grid_selection_rank'][:2] = [0, 0]
        a['grid_start_s'][:2] = 30
        a['grid_duration_s'][:2] = 120
    amend_actions(tmp_path, 'a', change)
    with pytest.raises(ValueError, match='teacher labels'):
        list(DatasetReader(tmp_path).iter_batches('train'))
