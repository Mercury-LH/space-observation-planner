"""V2 integrity tests use synthetic report fixtures, not new physical evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from obsplan.config import Config
from obsplan.data import load
from obsplan.schedule import Schedule, evaluate
from obsplan.training_data import v2_audit as v2
from obsplan.training_data.audit import AuditError, sha256
from obsplan.training_data.build import file_manifest
from obsplan.training_data.catalog import dump_json, generate_catalog
from obsplan.visibility import Windows
from test_training_audit import corpus as v1_fixture, edit_archive, repack


def report(rows):
    return dict(passed=len(rows) == 600, specs_sha256=hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest(), counts={"catalogs": len(rows)})


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = v1_fixture.__wrapped__(tmp_path)
    old, cid = 'cat-000', 'v2-cat-000'
    (root / 'catalogs' / old).rename(root / 'catalogs' / cid)
    (root / 'scenes' / (old+'-v0')).rename(root / 'scenes' / (cid+'-v0'))
    manifest = json.loads((root / 'manifest.json').read_text())
    cat = manifest['catalogs'][0]
    cat.update(catalog_id=cid, distribution='nominal_gaussian', n_stations=10, size_bin='small')
    scene = manifest['scenes'][0]
    scene.update(catalog_id=cid, scene_id=cid+'-v0', path='scenes/'+cid+'-v0',
                 distribution='nominal_gaussian', variant=0)
    dump_json(root / 'catalogs' / cid / 'catalog.json', cat)
    dump_json(root / scene['path'] / 'scene.json', scene)
    for filename in ('grid_plan.npz', 'plan.npz'):
        edit_archive(root / scene['path'] / filename,
                     lambda a: a.update(source_grid_row=np.arange(2, dtype=np.int32),
                                        candidate_id=np.array([0, 5], np.int32)))
    spec = {k:cat[k] for k in ('catalog_id','split','n_targets','n_stations','seed','start_utc','distribution','size_bin')}
    manifest.update(dataset_version=v2.VERSION, sampling_report=report([spec]))
    manifest['identity'].update(dataset_version=v2.VERSION, catalog_specs=[spec])
    dump_json(root / 'manifest.json', manifest)
    repack(root)
    monkeypatch.setattr(v2, '_sampling_report', report)
    return root


@pytest.mark.parametrize('stations', [4, 10, 16])
def test_pilot_streaming_audit_accepts_variable_station_counts(corpus, stations):
    manifest = json.loads((corpus / 'manifest.json').read_text())
    cat, scene, spec = manifest['catalogs'][0], manifest['scenes'][0], manifest['identity']['catalog_specs'][0]
    cpath, spath = corpus / 'catalogs' / cat['catalog_id'], corpus / scene['path']
    def change(a):
        a['stations'] = a['stations'][np.arange(stations) % 10]
    edit_archive(cpath / 'targets.npz', change)
    with np.load(cpath / 'targets.npz') as table:
        coords = table['stations']
    (spath / 'input/站址.txt').write_text('\n'.join(
        f'{j+1}: {x:.10f}, {y:.10f}, {z:.6f}' for j,(x,y,z) in enumerate(coords))+'\n')
    for path in (cpath/'windows.npz', spath/'pool_windows.npz'):
        win = Windows.load(path, 10)
        win.dark = [[(0,100)]] * stations
        win.save(path)
    cat['n_stations'] = scene['n_stations'] = spec['n_stations'] = stations
    scene['base_station_indices'] = list(range(stations))
    cfg=Config(**scene['config']).with_(data_dir=str(spath/'input'))
    with np.load(spath/'plan.npz') as p:
        plan=Schedule(*(p[k] for k in ('station','target','start_s','dur_s')))
    scene['metrics']=scene['grid_metrics']=evaluate(cfg,load(cfg),plan)
    manifest['sampling_report'] = report([spec])
    dump_json(cpath/'catalog.json', cat); dump_json(spath/'scene.json', scene)
    dump_json(corpus/'manifest.json', manifest); repack(corpus)
    result = v2.audit_dataset(corpus, allow_pilot=True)
    assert result['passed'] and result['pilot']
    assert result['plan_provenance']['matched_final_tasks'] == 2
    assert result['v1_reference']['checked'] is False
    assert result['counts']['scenes'] == 1
    with pytest.raises(AuditError, match='pilot needs explicit'):
        v2.audit_dataset(corpus)


@pytest.mark.parametrize('filename,change,match', [
    ('plan.npz', lambda a:a['source_grid_row'].__setitem__(0,1), 'source_grid_row'),
    ('plan.npz', lambda a:a['candidate_id'].__setitem__(0,5), 'task identity'),
    ('plan.npz', lambda a:a.pop('candidate_id'), 'candidate_id'),
    ('grid_plan.npz', lambda a:a['source_grid_row'].__setitem__(1,2), 'source_grid_row'),
])
def test_provenance_rejects_rehashed_corruption(corpus, filename, change, match):
    edit_archive(corpus/'scenes/v2-cat-000-v0'/filename, change)
    repack(corpus)
    with pytest.raises(AuditError, match=match):
        v2.audit_dataset(corpus, allow_pilot=True)


def test_sampling_report_and_spec_metadata_are_bound(corpus):
    manifest = json.loads((corpus/'manifest.json').read_text())
    manifest['sampling_report']['passed'] = True
    dump_json(corpus/'manifest.json', manifest)
    with pytest.raises(AuditError, match='recorded sampling report differs'):
        v2.audit_dataset(corpus, allow_pilot=True)
    manifest['sampling_report'] = report(manifest['identity']['catalog_specs'])
    manifest['identity']['catalog_specs'][0]['seed'] += 1
    manifest['sampling_report'] = report(manifest['identity']['catalog_specs'])
    dump_json(corpus/'manifest.json', manifest)
    with pytest.raises(AuditError, match='spec.seed'):
        v2.audit_dataset(corpus, allow_pilot=True)


def test_spec_fingerprint_uses_sampler_canonical_serialization(corpus):
    from obsplan.training_data.v2_sampling import specs_sha256
    manifest=json.loads((corpus/'manifest.json').read_text())
    rows=manifest['identity']['catalog_specs']
    manifest['identity']['specs_sha256']=specs_sha256(rows)
    # Compact canonical JSON is intentionally distinct from the transaction
    # identity hash serialization, which keeps default separator whitespace.
    assert manifest['identity']['specs_sha256'] != hashlib.sha256(
        json.dumps(rows,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    assert v2.audit_spec_contract(manifest)[0] == rows
    manifest['identity']['specs_sha256']='0'*64
    with pytest.raises(AuditError,match='frozen spec fingerprint differs'):
        v2.audit_spec_contract(manifest)


def test_full_contract_requires_exact_design_not_just_counts(monkeypatch):
    rows = [dict(catalog_id=f'v2-cat-{k:03d}', split='train' if k<420 else ('validation' if k<510 else 'test'),
                 n_targets=1000, n_stations=4, seed=k,
                 distribution='stress' if 480<=k<510 or 570<=k<600 else 'nominal_gaussian') for k in range(600)]
    monkeypatch.setattr(v2, '_canonical_specs', lambda:deepcopy(rows))
    monkeypatch.setattr(v2, '_sampling_report', report)
    m=dict(dataset_version=v2.VERSION, identity=dict(pilot=False,window_budget=12,catalog_specs=deepcopy(rows)),
           sampling_report=report(rows))
    assert len(v2.audit_spec_contract(m)[0]) == 600
    m['identity']['catalog_specs'][0]['seed'] = 999999
    m['sampling_report'] = report(m['identity']['catalog_specs'])
    with pytest.raises(AuditError, match='frozen v0.2 design'):
        v2.audit_spec_contract(m)


def test_sqlite_owner_detects_duplicates_and_closes(tmp_path):
    owner = v2.SQLiteOrbitOwner(tmp_path/'orbit.sqlite')
    h='ab'*32
    try:
        assert h not in owner and owner.get(h) is None
        owner[h]=('first',42)
        assert h in owner and owner.get(h)==('first',42)
        with pytest.raises(AuditError, match='duplicate orbital'):
            owner[h]=('second',0)
    finally:
        owner.close()


def make_reference(tmp_path):
    root=tmp_path/'v1-reference'; root.mkdir()
    catalogs=[]
    for i in range(30):
        spec=dict(catalog_id=f'cat-{i:03d}',split='train',n_targets=2,seed=2026110700+i,
                  start_utc='2026-01-10T00:00:00+00:00')
        _, table=generate_catalog(spec)
        p=root/'catalogs'/spec['catalog_id'];p.mkdir(parents=True)
        np.savez_compressed(p/'targets.npz',**table)
        catalogs.append(spec)
    m=dict(dataset_version='synthetic-v0.1',status='complete',identity=dict(pilot=False),
           catalogs=catalogs,files=file_manifest(root))
    dump_json(root/'manifest.json',m)
    return root


def test_reference_checksum_and_cross_version_orbit_reuse(corpus,tmp_path):
    reference=make_reference(tmp_path)
    m=json.loads((corpus/'manifest.json').read_text())
    m['identity']['v1_manifest_sha256']=sha256(reference/'manifest.json')
    dump_json(corpus/'manifest.json',m)
    with pytest.raises(AuditError,match='duplicate orbital parameters'):
        v2.audit_dataset(corpus,allow_pilot=True,v1_root=reference)
    (reference/'catalogs/cat-001/targets.npz').write_bytes(b'corrupted')
    with pytest.raises(AuditError,match='reference|checksum'):
        v2.audit_dataset(corpus,allow_pilot=True,v1_root=reference)


def test_retained_failure_payload_is_counted_and_hash_checked(corpus):
    manifest=json.loads((corpus/'manifest.json').read_text())
    spec=manifest['identity']['catalog_specs'][0]
    def h(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    failure=dict(type='KeyboardInterrupt',catalog_id='v2-cat-000',scene_id='v2-cat-000-v0',seed=spec['seed'],
                 attempt=1,spec_sha256=h(spec),identity_sha256=h(manifest['identity']),time_utc='2026-11-07T10:00:00+00:00')
    p=corpus/'failures/v2-cat-000/attempt-01/failure.json'
    dump_json(p,failure)
    manifest['failures']=[{**failure,'path':'failures/v2-cat-000/attempt-01'}]
    dump_json(corpus/'manifest.json',manifest)
    repack(corpus)
    result=v2.audit_dataset(corpus,allow_pilot=True)
    assert result['events']['retained_attempts']==1 and result['events']['retained_files']==1
    assert result['events']['failures']==1 and result['events']['ledger_entries_without_event']==1
    p.write_text('mutated')
    with pytest.raises(AuditError,match='byte size mismatch|SHA-256 mismatch'):
        v2.audit_dataset(corpus,allow_pilot=True)


def test_output_is_new_and_external(corpus,tmp_path,capsys):
    out=tmp_path/'v2-audit.json'
    v2.main([str(corpus),'--allow-pilot','--output',str(out)])
    assert json.loads(out.read_text())['passed']
    capsys.readouterr()
    with pytest.raises(AuditError,match='already exists'):
        v2.main([str(corpus),'--allow-pilot','--output',str(out)])
    with pytest.raises(AuditError,match='outside dataset root'):
        v2.main([str(corpus),'--allow-pilot','--output',str(corpus/'audit.json')])
