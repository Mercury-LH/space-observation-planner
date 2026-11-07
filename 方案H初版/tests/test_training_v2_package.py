"""Small opaque archives test v2 packaging; no fixture claims physical validity."""
from copy import deepcopy
import json
import tarfile

import pytest

from obsplan.training_data import v2_package as pack
from obsplan.training_data.audit import audit_files
from obsplan.training_data.download import digest, download
from test_training_package import write_json, refresh_manifest, rewrite_tar


@pytest.fixture
def corpus(tmp_path,monkeypatch):
    # Two groups exercise grouping independently of the frozen 600-catalog sampler.
    monkeypatch.setattr(pack,'CATALOG_COUNT',20)
    root,docs,output=tmp_path/'source',tmp_path/'docs',tmp_path/'staged'
    cats=[];scenes=[]
    for i in range(20):
        cid=f'v2-cat-{i:03d}'
        cat=dict(catalog_id=cid,split='train' if i<10 else 'test',n_targets=3,n_stations=4)
        cats.append(cat)
        cpath=root/'catalogs'/cid
        write_json(cpath/'catalog.json',cat)
        (cpath/'targets.npz').write_bytes(f'opaque targets {i}'.encode())
        (cpath/'windows.npz').write_bytes(f'opaque windows {i}'.encode())
        scene=dict(catalog_id=cid,scene_id=f'{cid}-v0',path=f'scenes/{cid}-v0',split=cat['split'],
                   variant=0,legal=True,geometry_checked=True,grid_tasks=2,final_tasks=2)
        scenes.append(scene)
        spath=root/scene['path'];write_json(spath/'scene.json',scene)
        for name in ('actions.npz','grid_plan.npz','plan.npz','pool_windows.npz','pool_source_indices.npz',
                     'grid_geometry.npz','final_geometry.npz'):
            (spath/name).write_bytes(f'opaque {i}/{name}'.encode())
        check=dict(legal=True,geometry_checked=True,step_s=1.,tasks=2,violations={})
        write_json(spath/'validation.json',dict(grid=check,final=check))
        for name in ('轨道.txt','站址.txt','低优先级目标.txt','中优先级目标.txt','高优先级目标.txt'):
            p=spath/'input'/name;p.parent.mkdir(exist_ok=True);p.write_text(f'input {i}\n')
    failure=root/'failures/v2-cat-002/attempt-01/partial.bin'
    failure.parent.mkdir(parents=True);failure.write_bytes(b'retained interrupted attempt')
    (root/'events.jsonl').write_text('{"kind":"explicit_test_fixture"}\n')
    m=dict(dataset_version=pack.VERSION,status='building',identity=dict(pilot=False,catalog_specs=deepcopy(cats)),
           catalogs=cats,scenes=scenes,summary=dict(catalogs=20,scenes=20))
    refresh_manifest(root,m);docs.mkdir()
    return root,docs,output,m


def finish(root,docs,m):
    m['status']='complete';refresh_manifest(root,m)
    write_json(docs/'audit.json',dict(passed=True,pilot=False,dataset_version=pack.VERSION,counts=m['summary'],
              v1_reference=dict(checked=True),sampling_report=dict(passed=True),plan_provenance=dict(passed=True),
              checksums=dict(manifest_sha256=digest(root/'manifest.json'),events_sha256=digest(root/'events.jsonl'))))


def test_production_group_boundaries():
    assert pack.group_name(0)=='synthetic-v0.2-group-000.tar.gz'
    assert pack.group_name(59)=='synthetic-v0.2-group-059.tar.gz'
    for value in (-1,60,True,1.5):
        with pytest.raises(ValueError,match='invalid group'):
            pack.group_name(value)


def test_group_staging_is_fixed_and_contains_retained_failures(corpus):
    root,_,out,m=corpus
    asset=pack.stage_group(root,out,0)
    assert asset['catalog_ids']==[f'v2-cat-{i:03d}' for i in range(10)]
    assert asset['splits']==['train'] and asset['group_id']==0
    with tarfile.open(out/asset['name']) as tar:
        names=tar.getnames()
        assert 'failures/v2-cat-002/attempt-01/partial.bin' in names
        assert not any('v2-cat-010' in p for p in names)
    with pytest.raises(FileExistsError):pack.stage_group(root,out,0)


@pytest.mark.parametrize('fault',['missing_scene','missing_catalog','wrong_spec','missing_input','bad_validation','pilot'])
def test_stage_rejects_incomplete_or_changed_catalog_group(corpus,fault):
    root,_,out,m=corpus
    if fault=='missing_scene':m['scenes'].pop(0)
    elif fault=='missing_catalog':m['catalogs'].pop(0)
    elif fault=='wrong_spec':m['identity']['catalog_specs'][0]['n_targets']=5
    elif fault=='missing_input':(root/'scenes/v2-cat-000-v0/input/站址.txt').unlink()
    elif fault=='bad_validation':
        p=root/'scenes/v2-cat-000-v0/validation.json';r=json.loads(p.read_text());r['grid']['step_s']=30;write_json(p,r)
    else:m['identity']['pilot']=True
    refresh_manifest(root,m)
    with pytest.raises(ValueError):pack.stage_group(root,out,0)
    assert not out.exists()


@pytest.mark.parametrize('fault',['bytes','duplicate','missing'])
def test_reuse_checks_actual_archive_members(corpus,fault):
    root,docs,out,m=corpus
    asset=pack.stage_group(root,out,0)
    def change(entries):
        if fault=='bytes':
            member,body=entries[0];entries[0]=(member,bytes([body[0]^1])+body[1:])
        elif fault=='duplicate':entries.append(deepcopy(entries[0]))
        else:entries.pop()
    rewrite_tar(out/asset['name'],change);finish(root,docs,m)
    with pytest.raises(ValueError):pack.package(root,out,docs,reuse_shards=True)
    assert not (out/pack.INDEX_NAME).exists()


def test_staged_and_fresh_packages_download_to_complete_immutable_payload(corpus,tmp_path,monkeypatch):
    root,docs,out,m=corpus
    asset=pack.stage_group(root,out,0);before=digest(out/asset['name'])
    finish(root,docs,m)
    result=pack.package(root,out,docs,reuse_shards=True)
    assert digest(out/asset['name'])==before
    assert len(result['assets'])==3 and result['group_size']==10
    fresh=pack.package(root,tmp_path/'fresh',docs)
    assert result==fresh
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('verified cache needs no network'))
    dest=tmp_path/'extracted'
    download(out/pack.INDEX_NAME,dest,out)
    audited=audit_files(dest.resolve(),json.loads((dest/'manifest.json').read_text()))
    assert audited['manifest_sha256']==result['manifest_sha256']
    assert (dest/'failures/v2-cat-002/attempt-01/partial.bin').read_bytes()==b'retained interrupted attempt'
    assert not (dest/'audit.json').exists()
    with pytest.raises(FileExistsError,match='final package metadata'):
        pack.package(root,out,docs,reuse_shards=True)


@pytest.mark.parametrize('fault',['v1_unchecked','sampling_failed','provenance_failed','stale_events','counts','unexpected_root_file'])
def test_final_requires_complete_current_audit_and_archive_coverage(corpus,fault):
    root,docs,out,m=corpus;finish(root,docs,m)
    report=json.loads((docs/'audit.json').read_text())
    if fault=='v1_unchecked':report['v1_reference']['checked']=False
    elif fault=='sampling_failed':report['sampling_report']['passed']=False
    elif fault=='provenance_failed':report['plan_provenance']['passed']=False
    elif fault=='stale_events':(root/'events.jsonl').write_text('changed')
    elif fault=='counts':report['counts']={}
    else:
        (root/'unlisted-contract-file.txt').write_text('not owned by any group')
        refresh_manifest(root,m);report['checksums']['manifest_sha256']=digest(root/'manifest.json')
    write_json(docs/'audit.json',report)
    with pytest.raises(ValueError):pack.package(root,out,docs)
    assert not (out/pack.INDEX_NAME).exists()


def test_no_staging_pollution(corpus):
    root,_,_,_=corpus
    with pytest.raises(ValueError,match='outside the immutable'):
        pack.stage_group(root,root/'staging',0)
