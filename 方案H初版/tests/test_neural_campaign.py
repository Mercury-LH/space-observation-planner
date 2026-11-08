import argparse
import json
from pathlib import Path
import subprocess

import pytest

from obsplan.neural_lns import campaign
from obsplan.neural_lns.runtime import load_scene, write_new


def test_worker_outer_timeout_is_retained(tmp_path, monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(a[0], k['timeout'])
    monkeypatch.setattr(campaign.subprocess, 'run', timeout)
    request=dict(catalog_id='c',seed=2026,method='rule',budget_s=5,
                 calibration_s=[1,1,1],freeze_hash='f')
    result=campaign.isolated_run(request,tmp_path/'attempt',0.001)
    assert result['status']=='timeout'
    assert json.loads((tmp_path/'attempt/result.json').read_text())==result
    assert (tmp_path/'attempt/request.json').is_file()


def test_holdout_loading_requires_explicit_split_before_input_access(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    write_new(root/'manifest.json',{'dataset_version':'synthetic-v0.2','scenes':[
        {'catalog_id':'test-cat','scene_id':'test-cat-v0','split':'test','path':'scenes/test-cat-v0'}]})
    with pytest.raises(ValueError,match='forbidden split'):
        load_scene(root,'test-cat',tmp_path/'cache')


def _campaign_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign,'environment',lambda *a:{'environment_hash':'env'})
    monkeypatch.setattr(campaign,'source_hash',lambda:'code')
    root=tmp_path/'dataset';root.mkdir();write_new(root/'manifest.json',{})
    calibration={'complete':True,'code_hash':'code','environment_hash':'env','workers':1,
                 'freeze_hash':'f','budgets':{'c':{'budget_s':5,'calibration_s':[1,1,1],'finish_reserve_s':.1}}}
    cp=tmp_path/'calibration.json';write_new(cp,calibration)
    args=argparse.Namespace(dataset_root=str(root),output=str(tmp_path/'out'),cache_dir=str(tmp_path/'cache'),
            instance_id='test-instance',workers=1,selection='pilot',calibration=str(cp),models=None,
            methods='rule',runtime_freeze=None,device='cpu',wall_limit_s=2,resume_campaign=False,
            application_data_dir=None)
    doc={'freeze':{'freeze_hash':'f'},'protocol':{'methods':['rule'],'model_methods':[], 'seeds':[2026]},
         'splits':{'pilot_validation':['c']}}
    return args,doc


def test_campaign_resume_preserves_failed_attempt_and_does_not_rerun(tmp_path,monkeypatch):
    args,doc=_campaign_fixture(tmp_path,monkeypatch)
    calls=[]
    def fail(req,out,wall):
        calls.append(req)
        out.mkdir(parents=True)
        row={**req,'status':'failed','error':'deliberate worker failure'}
        write_new(out/'result.json',row)
        return row
    monkeypatch.setattr(campaign,'isolated_run',fail)
    first=campaign.evaluate_campaign(args,doc)
    args.resume_campaign=True
    second=campaign.evaluate_campaign(args,doc)
    assert len(calls)==1
    assert json.loads(Path(first['records']).read_text())==json.loads(Path(second['records']).read_text())
    assert json.loads(Path(second['records']).read_text())[0]['status']=='failed'


def test_changed_code_refuses_calibration_before_creating_output(tmp_path,monkeypatch):
    args,doc=_campaign_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(campaign,'source_hash',lambda:'changed')
    with pytest.raises(ValueError,match='changed'):
        campaign.evaluate_campaign(args,doc)
    assert not Path(args.output).exists()


def test_resume_cannot_switch_inference_device(tmp_path,monkeypatch):
    args,doc=_campaign_fixture(tmp_path,monkeypatch)
    def finish(req,out,wall):
        out.mkdir(parents=True)
        row={**req,'status':'completed'};write_new(out/'result.json',row);return row
    monkeypatch.setattr(campaign,'isolated_run',finish)
    campaign.evaluate_campaign(args,doc)
    args.resume_campaign=True;args.device='cuda'
    with pytest.raises(ValueError,match='resume identity'):
        campaign.evaluate_campaign(args,doc)


def test_formal_claim_is_bound_to_identity_not_freeze_filename(tmp_path):
    identity={'models':{'neural':'sha'},'code_hash':'fixed'}
    registry=tmp_path/'registry'
    campaign.claim_formal_campaign(identity,tmp_path/'first',registry=registry)
    with pytest.raises(ValueError,match='already assigned'):
        campaign.claim_formal_campaign(dict(identity),tmp_path/'second',registry=registry)
    campaign.claim_formal_campaign(identity,tmp_path/'first',resume=True,registry=registry)
