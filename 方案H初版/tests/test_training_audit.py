"""Audit contract tests; fixture geometry reports are synthetic, not Skyfield evidence.

The real producer's pilot is audited separately. These small fixtures exercise
corruption and consistency checks without repeating expensive orbital geometry.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest
from sgp4.api import Satrec
from sgp4.exporter import export_tle

from obsplan.config import Config
from obsplan.famou_h_eval import FEATURES
from obsplan.schedule import Schedule, evaluate
from obsplan.training_data.audit import AuditError, audit_dataset, audit_orbits, main
from obsplan.training_data.build import file_manifest, save_plan
from obsplan.training_data.catalog import dump_json, generate_catalog, write_native_input
from obsplan.visibility import Windows


def repack(root):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files'] = file_manifest(root)
    dump_json(path, manifest)


def edit_archive(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    change(arrays)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    spec = dict(catalog_id='cat-000', split='train', n_targets=2, seed=2026110700,
                start_utc='2026-01-10T00:00:00+00:00', distribution='nominal')
    ds, table = generate_catalog(spec)
    cat_path = root / 'catalogs' / 'cat-000'
    cat_path.mkdir(parents=True)
    np.savez_compressed(cat_path / 'targets.npz', **table)
    win = Windows(np.array([0, 0], np.int16), np.array([0, 1], np.int32),
                  np.array([0, 10], np.int32), np.array([8, 18], np.int32),
                  [[(0, 100)]] * 10, 17281, np.zeros(2, bool))
    win.save(cat_path / 'windows.npz')
    cat = dict(**spec, n_windows=2, sgp4_failed=0, geometry_wall_s=.1)
    dump_json(cat_path / 'catalog.json', cat)
    scene_path = root / 'scenes' / 'cat-000-v0'
    scene_path.mkdir(parents=True)
    write_native_input(scene_path / 'input', ds)
    win.save(scene_path / 'pool_windows.npz')
    np.savez_compressed(scene_path / 'pool_source_indices.npz', catalog_window_id=np.array([0, 1], np.int32))
    cw = np.repeat([0, 1], 5).astype(np.int32)
    cm = np.tile(np.arange(4, 9), 2).astype(np.int16)
    selected = np.zeros(10, bool)
    selected[[0, 5]] = True
    rank = np.full(10, -1, np.int32)
    rank[[0, 5]] = [0, 1]
    starts = np.full(10, -1, np.int32)
    starts[[0, 5]] = [0, 300]
    durations = np.where(selected, 120, -1).astype(np.int32)
    np.savez_compressed(scene_path / 'actions.npz', features=np.ones((10, 20), np.float32),
                        candidate_window=cw, candidate_duration_cells=cm,
                        rule_score_uncovered=np.ones(10, np.float32), rule_score_covered=np.ones(10, np.float32),
                        grid_selected=selected, grid_selection_rank=rank, grid_start_s=starts,
                        grid_duration_s=durations, grid_selected_count=selected.astype(np.int32),
                        grid_task_candidate=np.array([0, 5], np.int32))
    plan = Schedule(np.array([0, 0]), np.array([0, 1]), np.array([0, 300]), np.array([120, 120]))
    save_plan(scene_path / 'grid_plan.npz', plan, candidate_id=np.array([0, 5], np.int32))
    save_plan(scene_path / 'plan.npz', plan)
    for name in ('grid', 'final'):
        np.savez_compressed(scene_path / f'{name}_geometry.npz',
                            per_task=np.array([[20., 30., -20.]] * 2), invalid_mask=np.zeros(2, bool))
    report = dict(legal=True, geometry_checked=True, step_s=1., violations={}, tasks=2,
                  original_tasks=2, removed_tasks=0, failed_attempts=[], min_elev_margin=10.,
                  max_elev_margin=55., sun_margin=14., wall_s=.1)
    dump_json(scene_path / 'validation.json', dict(grid=report, final=report))
    cfg = Config(start_utc=spec['start_utc'])
    config = asdict(cfg)
    config['data_dir'] = 'input'
    config.pop('cache_dir')
    scene = dict(scene_id='cat-000-v0', catalog_id='cat-000', split='train', path='scenes/cat-000-v0',
                 distribution='nominal', config=config, n_targets=2, n_stations=10,
                 base_station_indices=list(range(10)), pool_windows=2, full_variant_windows=2,
                 candidate_rows=10, grid_positive_candidates=2, grid_tasks=2, final_tasks=2,
                 metrics=evaluate(cfg, ds, plan), grid_metrics=evaluate(cfg, ds, plan),
                 repairs=0, legal=True, geometry_checked=True, teacher_timings=dict(total_s=.1),
                 validation_wall_s=.2, scene_wall_s=.4)
    dump_json(scene_path / 'scene.json', scene)
    manifest = dict(identity=dict(pilot=True, window_budget=12), status='complete', feature_names=FEATURES,
                    catalogs=[cat], scenes=[scene], summary=dict(catalogs=1, scenes=1,
                    unique_synthetic_targets=2, target_instances=2, candidate_rows=10,
                    grid_tasks=2, final_tasks=2, repaired_tasks=0))
    dump_json(root / 'manifest.json', manifest)
    events = [dict(kind='build_start', time_utc='2026-11-07T10:00:00+00:00'),
              dict(kind='build_complete', time_utc='2026-11-07T10:00:01+00:00')]
    (root / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    repack(root)
    return root


def test_complete_pilot_audit_and_no_implicit_threshold_bypass(corpus):
    result = audit_dataset(corpus, allow_pilot=True)
    assert result['passed'] and result['counts']['candidate_rows'] == 10
    assert result['counts']['grid_tasks'] == result['counts']['final_tasks'] == 2
    assert result['events']['failures'] == 0
    assert result['groups']['split']['train']['scenes'] == 1
    with pytest.raises(AuditError, match='pilot needs explicit'):
        audit_dataset(corpus)
    manifest = json.loads((corpus / 'manifest.json').read_text())
    manifest['identity']['pilot'] = False
    dump_json(corpus / 'manifest.json', manifest)
    with pytest.raises(AuditError, match='size thresholds'):
        audit_dataset(corpus, allow_pilot=True)


def test_checksum_and_unlisted_file_rejected(corpus):
    path = corpus / 'scenes/cat-000-v0/input/站址.txt'
    original = path.read_bytes()
    path.write_bytes(original+b'\n')
    with pytest.raises(AuditError, match='byte size mismatch'):
        audit_dataset(corpus, allow_pilot=True)
    path.write_bytes(original)
    (corpus / 'unlisted.txt').write_text('unexpected')
    with pytest.raises(AuditError, match='incomplete file manifest'):
        audit_dataset(corpus, allow_pilot=True)


def test_tle_checksum_rejected_even_if_manifest_is_rehashed(corpus):
    def change(a):
        line = str(a['line1'][0])
        a['line1'][0] = line[:-1]+str((int(line[-1])+1) % 10)
    edit_archive(corpus / 'catalogs/cat-000/targets.npz', change)
    repack(corpus)
    with pytest.raises(AuditError, match='TLE checksum'):
        audit_dataset(corpus, allow_pilot=True)


def test_orbit_identity_ignores_renamed_satellite_numbers():
    spec = dict(catalog_id='a', split='train', n_targets=2, seed=2026110700,
                start_utc='2026-01-10T00:00:00+00:00')
    _, table = generate_catalog(spec)
    owner = {}
    audit_orbits(table, spec, owner)
    renamed = {key: value.copy() for key, value in table.items()}
    for i in range(2):
        sat = Satrec.twoline2rv(str(table['line1'][i]), str(table['line2'][i]))
        # Exported TLEs differ only in their identity fields.
        from sgp4.api import WGS72
        other = Satrec()
        other.sgp4init(WGS72, 'i', 20001+i, sat.jdsatepoch+sat.jdsatepochF-2433281.5,
                       sat.bstar, sat.ndot, sat.nddot, sat.ecco, sat.argpo, sat.inclo,
                       sat.mo, sat.no_kozai, sat.nodeo)
        renamed['line1'][i], renamed['line2'][i] = export_tle(other)
        renamed['satnum'][i] = 20001+i
    with pytest.raises(AuditError, match='duplicate orbital parameters'):
        audit_orbits(renamed, {**spec, 'catalog_id': 'b'}, owner)


@pytest.mark.parametrize('name,change,match', [
    ('pool_source_indices.npz', lambda a: a['catalog_window_id'].__setitem__(0, 1), 'window mapping'),
    ('pool_windows.npz', lambda a: a['target'].__setitem__(0, 1), 'pool/catalog window mismatch'),
    ('actions.npz', lambda a: a['candidate_window'].__setitem__(1, 99), 'candidate window indices'),
    ('actions.npz', lambda a: a['grid_start_s'].__setitem__(0, 30), 'first-selection times'),
    ('grid_plan.npz', lambda a: a['candidate_id'].__setitem__(0, 1), 'task candidate mapping'),
    ('final_geometry.npz', lambda a: a['per_task'].__setitem__((0, 0), 9.), 'geometry extrema'),
    ('final_geometry.npz', lambda a: a['invalid_mask'].__setitem__(0, True), 'failed geometry evidence'),
])
def test_rehashed_semantic_corruption_rejected(corpus, name, change, match):
    edit_archive(corpus / 'scenes/cat-000-v0' / name, change)
    repack(corpus)
    with pytest.raises(AuditError, match=match):
        audit_dataset(corpus, allow_pilot=True)


def test_false_one_second_claim_rejected(corpus):
    path = corpus / 'scenes/cat-000-v0/validation.json'
    report = json.loads(path.read_text())
    report['final']['step_s'] = 30.
    dump_json(path, report)
    repack(corpus)
    with pytest.raises(AuditError, match='one-second evidence'):
        audit_dataset(corpus, allow_pilot=True)


def test_score_recomputed_instead_of_trusting_matching_jsons(corpus):
    path = corpus / 'scenes/cat-000-v0/scene.json'
    scene = json.loads(path.read_text())
    scene['metrics']['score'] += 1
    dump_json(path, scene)
    manifest = json.loads((corpus / 'manifest.json').read_text())
    manifest['scenes'][0] = scene
    dump_json(corpus / 'manifest.json', manifest)
    repack(corpus)
    with pytest.raises(AuditError, match=r'metrics.score: value mismatch'):
        audit_dataset(corpus, allow_pilot=True)


def test_cli_writes_only_new_external_output(corpus, capsys):
    output = corpus.parent / 'audit.json'
    main([str(corpus), '--allow-pilot', '--output', str(output)])
    assert json.loads(output.read_text())['passed']
    assert json.loads(capsys.readouterr().out)['passed']
    with pytest.raises(AuditError, match='already exists'):
        main([str(corpus), '--allow-pilot', '--output', str(output)])
    with pytest.raises(AuditError, match='outside dataset root'):
        main([str(corpus), '--allow-pilot', '--output', str(corpus / 'audit.json')])
