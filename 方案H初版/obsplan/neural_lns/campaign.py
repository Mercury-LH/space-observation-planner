"""Serial, append-only scene runs; calibration and model freeze precede test access."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback
import uuid

from .runtime import canonical_hash, environment, file_hash, load_scene, source_hash, write_new
from .protocol import calibrated_budget, load_frozen_protocol, load_runtime_freeze
from .known import CATALOG_ID


def claim_formal_campaign(identity, campaign_directory, *, resume=False, registry=None):
    """Bind the semantic frozen model/config identity, independent of JSON filename."""
    registry = Path(registry) if registry else Path(__file__).resolve().parents[2]/'outputs/neural-campaign-registry'
    claim_path = registry/(canonical_hash(identity)+'.json')
    claim = {'identity':identity,'output':str(Path(campaign_directory).resolve())}
    if claim_path.exists():
        if not resume or json.loads(claim_path.read_text()) != claim:
            raise ValueError('frozen test model set already assigned to a campaign; only resume unstarted rows')
    else:
        write_new(claim_path,claim)
    return claim_path


def worker(request_path):
    request_path = Path(request_path)
    req = json.loads(request_path.read_text())
    out = request_path.parent
    started = time.perf_counter()
    cpu_started = time.process_time()
    env = environment(req.get('instance_id'))
    row = {k: req[k] for k in ('catalog_id', 'seed', 'method', 'budget_s', 'calibration_s', 'freeze_hash')}
    row.update(status='failed', environment_hash=env['environment_hash'], environment=env,
               candidate_generator_hash=file_hash(Path(__file__).with_name('search.py')),
               code_hash=source_hash(), validation={'legal': False, 'engine': 'skyfield',
                                                  'step_s': 1, 'endpoints': True})
    row['cache_policy'] = {'window_cache': 'published full windows or isolated application cache',
            'warmup': 'no in-process warmup; native Numba disk cache may exist',
            'numba_disk_cache_present': any(Path(__file__).resolve().parents[1].rglob('*.nbc')),
            'comparison_execution':'serial fresh workers'}
    try:
        import numba
        import numpy as np
        from obsplan.benchmark_worker import check_schedule
        from obsplan.portfolio import solve as original_solve
        from obsplan.schedule import evaluate
        from obsplan.validator import validate
        from obsplan.refine import near_windows
        from . import plugin
        # Public benchmark prepares short-window geometry outside solve timing.
        # Cache identity includes catalog, preventing same-date synthetic scenes
        # from sharing scenario-key-only short-window files.
        scene_cache = str(Path(req['cache_dir']) / req['catalog_id'])
        if req['catalog_id'] == CATALOG_ID:
            from .known import load_application
            pb, metadata = load_application(req['application_data_dir'], scene_cache)
        else:
            pb, metadata = load_scene(req['dataset_root'], req['catalog_id'], scene_cache,
                                      allowed_splits=tuple(req['allowed_splits']))
        numba.set_num_threads(req['workers'])
        row.update(metadata)
        row['data_identity'] = {k:metadata[k] for k in ('input_hash','config_hash','manifest_sha256','windows_sha256') if k in metadata}
        if req.get('data_identity') is not None and req['data_identity'] != row['data_identity']:
            raise ValueError('instance data differs from timing calibration')
        near_started = time.perf_counter()
        near_windows(pb)
        row['near_windows_prepare_s'] = time.perf_counter()-near_started
        solve_started = time.perf_counter()
        if req['method'] == 'calibration':
            schedule, info = original_solve(pb, {'preset': 'hplus_fast', 'h_iters': 300,
                    'h_fast': False, 'finish_near': True, 'workers': req['workers'], 'seed': 2026})
        else:
            mode = {'hplus_extended': 'original_hplus', 'h1_multi_bound': 'original_hplus',
                    'dynamic_random_init': 'neural'}.get(req['method'], req['method'])
            spec = {'enabled': True, 'mode': mode, 'budget_s': req['budget_s'],
                    'initial_h_iters': 300, 'initial_h_fast': False,
                    'workers': req['workers'], 'seed': req['seed'], 'finish_near': True,
                    'finish_reserve_s': req.get('finish_reserve_s', 0.05),
                    'device': req.get('device', 'cpu')}
            if req.get('checkpoint'):
                spec['checkpoint'] = req['checkpoint']
                row['model_hash'] = file_hash(req['checkpoint'])
            if req['method'] == 'h1_multi_bound':
                from benchmarks.h_acceleration_bounds import repair_coverage_multi_bound, _check_objective
                from obsplan import local_search
                _check_objective(pb)
                old = local_search.repair_coverage
                spec.update(initial_h_iters=1, initial_h_fast=True)
                try:
                    local_search.repair_coverage = repair_coverage_multi_bound
                    schedule, info = plugin.solve(pb, spec)
                finally:
                    local_search.repair_coverage = old
            else:
                schedule, info = plugin.solve(pb, spec)
        row['solve_s'] = time.perf_counter() - solve_started
        schedule = check_schedule(schedule, pb.ds)
        np.savez_compressed(out/'plan.npz', **{k: getattr(schedule, k) for k in
                                              ('station', 'target', 'start_s', 'dur_s')})
        write_new(out/'solver-info.json', info)
        row['initial_plan_hash'] = info.get('initial_plan_hash')
        row['initial_C'] = info.get('initial', {}).get('C')
        validation_started = time.perf_counter()
        report = validate(pb.cfg, pb.ds, schedule, step_s=1)
        row['validation_s'] = time.perf_counter()-validation_started
        row['validation'].update(legal=bool(report['legal'] and report['geometry_checked']),
                                  violations=report['violations'], geometry_checked=report['geometry_checked'])
        np.savez_compressed(out/'validation-geometry.npz', per_task=report['per_task'], invalid_mask=report['invalid_mask'])
        write_new(out/'validation.json', {k:v for k,v in report.items() if k not in ('per_task','invalid_mask')})
        row['validation']['report_sha256'] = file_hash(out/'validation.json')
        row['validation']['geometry_sha256'] = file_hash(out/'validation-geometry.npz')
        ev = evaluate(pb.cfg, pb.ds, schedule)
        row['metrics'] = {'C': ev['covered'], 'S': ev['score'], 'Q': ev['combined']}
        row['stage_timings_s'] = info.get('stage_timings_s', {})
        row['status'] = ('timeout' if req['method'] != 'calibration' and row['solve_s'] > req['budget_s']
                         else 'completed' if row['validation']['legal'] else 'invalid')
        row['plan_sha256'] = file_hash(out/'plan.npz')
    except BaseException as exc:
        row.update(error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    row['end_to_end_s'] = time.perf_counter()-started
    row['cpu_s'] = time.process_time()-cpu_started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row['peak_rss_bytes'] = int(rss if sys.platform=='darwin' else rss*1024)
    write_new(out/'result.json', row)
    return 0 if row['status'] == 'completed' else 1


def isolated_run(request, directory, wall_limit_s):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory/'request.json', request)
    started = time.perf_counter()
    with (directory/'worker.log').open('x') as log:
        try:
            result = subprocess.run([sys.executable, '-m', __name__, 'worker',
                                     str(directory/'request.json')], stdout=log, stderr=log,
                                    timeout=wall_limit_s)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
    path = directory/'result.json'
    if path.exists():
        row = json.loads(path.read_text())
    else:
        row = {k: request.get(k) for k in ('catalog_id', 'seed', 'method', 'budget_s', 'calibration_s', 'freeze_hash')}
        row.update(status='timeout' if code is None else 'failed', error='worker did not complete',
                   end_to_end_s=time.perf_counter()-started, worker_returncode=code)
        write_new(path, row)
    return row


def selection(document, name):
    splits = document['splits']
    if name == 'pilot':
        return splits['pilot_validation']
    if name == 'test':
        return splits['splits']['test']
    if name == 'application':
        return [CATALOG_ID]
    if name == 'final':
        return splits['splits']['test'] + [CATALOG_ID]
    if name == 'train40':
        return splits['train_subsets']['40']
    raise ValueError('unknown selection')


def calibrate(args, document):
    # Test calibration may measure baseline times only; it never selects a model.
    catalogs = selection(document, args.selection)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    rows, budgets = [], {}
    for cid in catalogs:
        timings, finishes, data_identities = [], [], []
        for repeat in range(3):
            req = base_request(args, document, cid, 2026, 'calibration')
            row = isolated_run(req, out/cid/f'calibration-{repeat}', args.wall_limit_s)
            rows.append(row)
            if row['status'] == 'completed':
                timings.append(row['solve_s'])
                finishes.append(row['stage_timings_s']['finish'])
                data_identities.append(row['data_identity'])
        same_data = len({canonical_hash(v) for v in data_identities}) == 1
        budgets[cid] = {'calibration_s': timings, 'budget_s': calibrated_budget(timings) if len(timings)==3 else None,
                        'finish_reserve_s': max(0.05, 1.25*statistics.median(finishes)) if len(finishes)==3 else None,
                        'data_identity':data_identities[0] if same_data else None}
        if not same_data:
            budgets[cid]['budget_s'] = None
    result = {'selection': args.selection, 'budgets': budgets, 'runs': rows,
              'code_hash': source_hash(), 'environment_hash': environment(args.instance_id)['environment_hash'],
              'freeze_hash': document['freeze']['freeze_hash'], 'workers': args.workers,
              'cache_policy': 'released complete windows; native numba disk cache; three fresh serial workers',
              'complete': all(v['budget_s'] is not None for v in budgets.values())}
    write_new(out/'calibration.json', result)
    return result


def base_request(args, document, cid, seed, method):
    return dict(catalog_id=cid, seed=seed, method=method, dataset_root=str(Path(args.dataset_root).resolve()),
                cache_dir=str(Path(args.cache_dir).resolve()), workers=args.workers,
                instance_id=args.instance_id, budget_s=None, calibration_s=[],
                freeze_hash=document['freeze']['freeze_hash'], allowed_splits=['train','validation','test'],
                application_data_dir=getattr(args,'application_data_dir',None),
                device=getattr(args, 'device', 'cpu'))


def evaluate_campaign(args, document):
    calibration = json.loads(Path(args.calibration).read_text())
    env_hash = environment(args.instance_id)['environment_hash']
    if (not calibration['complete'] or calibration['code_hash'] != source_hash()
            or calibration['environment_hash'] != env_hash or calibration['workers'] != args.workers
            or calibration['freeze_hash'] != document['freeze']['freeze_hash']):
        raise ValueError('calibration is incomplete or code/environment/protocol changed')
    models = json.loads(Path(args.models).read_text()) if args.models else {}
    if args.selection in ('test','application','final'):
        frozen = load_runtime_freeze(args.runtime_freeze, args.protocol) if args.runtime_freeze else {}
        required = {'models': {m:{s:file_hash(p) for s,p in values.items()} for m,values in models.items()},
                    'code_hash': source_hash(), 'calibration_hash': file_hash(args.calibration),
                    'protocol_hash': document['freeze']['freeze_hash'], 'environment_hash': env_hash,
                    'inference_device':args.device}
        if frozen.get('identity') != required or frozen.get('pilot_gate_passed') is not True:
            raise ValueError('test requires a complete pre-test runtime freeze and passed pilot gate')
        if set(models) != set(document['protocol']['model_methods']) or any(
                set(v) != {'2026','2027','2028'} for v in models.values()):
            raise ValueError('test requires all three seeds for every model method')
    methods = args.methods.split(',') if args.methods else document['protocol']['methods']
    seeds = [int(s) for s in args.seeds.split(',')] if getattr(args,'seeds',None) else document['protocol']['seeds']
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(document['protocol']['seeds']) or 2026 not in seeds:
        raise ValueError('pilot seeds must be a unique registered subset including 2026')
    if len(set(methods)) != len(methods) or not set(methods).issubset(document['protocol']['methods']):
        raise ValueError('duplicate or unknown comparison method')
    if args.selection in ('test','application','final') and methods != document['protocol']['methods']:
        raise ValueError('test campaign must contain all preregistered methods')
    if args.selection in ('test','application','final') and seeds != document['protocol']['seeds']:
        raise ValueError('formal campaign requires the three frozen seeds')
    out = Path(args.output).resolve()
    campaign_identity = {'selection':args.selection, 'methods':methods,'seeds':seeds,
            'device':args.device,'workers':args.workers,'wall_limit_s':args.wall_limit_s,
            'cache_dir':str(Path(args.cache_dir).resolve()),
            'application_data_dir':str(Path(args.application_data_dir).resolve()) if args.application_data_dir else None,
            'calibration_sha256':file_hash(args.calibration), 'code_hash':source_hash(),
            'environment_hash':env_hash,'freeze_hash':document['freeze']['freeze_hash'],
            'model_hashes':{m:{s:file_hash(p) for s,p in values.items()} for m,values in models.items()},
            'dataset_manifest_sha256':file_hash(Path(args.dataset_root)/'manifest.json')}
    if args.resume_campaign:
        if json.loads((out/'campaign.json').read_text()) != campaign_identity:
            raise ValueError('resume identity differs; existing attempts cannot be overwritten')
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_new(out/'campaign.json',campaign_identity)
    if args.selection in ('test','application','final'):
        claim_formal_campaign(required,out,resume=args.resume_campaign)
    rows = []
    for cid in selection(document, args.selection):
        for seed in seeds:
            for method in methods:
                req = base_request(args, document, cid, seed, method)
                req.update(calibration['budgets'][cid])
                if method in document['protocol']['model_methods']:
                    req['checkpoint'] = str(Path(models[method][str(seed)]).resolve())
                run_dir=out/cid/str(seed)/method
                if run_dir.exists():
                    if not args.resume_campaign:
                        raise FileExistsError(run_dir)
                    if (run_dir/'result.json').exists():
                        row=json.loads((run_dir/'result.json').read_text())
                    else:
                        row={k:req[k] for k in ('catalog_id','seed','method','budget_s','calibration_s','freeze_hash')}
                        row.update(status='interrupted',error='prior attempt interrupted; retained, not retried')
                        write_new(run_dir/'result.json',row)
                    rows.append(row)
                    continue
                row = isolated_run(req, run_dir, args.wall_limit_s)
                rows.append(row)
                with (out/'runs.jsonl').open('a') as ledger:
                    ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    result_path=out/(f'runs-resume-{uuid.uuid4().hex}.json' if args.resume_campaign else 'runs.json')
    write_new(result_path, rows)
    return {'runs': len(rows), 'output': str(out),'records':str(result_path)}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    w=sub.add_parser('worker'); w.add_argument('request')
    for command in ('calibrate','evaluate'):
        q=sub.add_parser(command)
        q.add_argument('--dataset-root',required=True);q.add_argument('--output',required=True)
        q.add_argument('--protocol',default='experiments/neural-lns-v1')
        q.add_argument('--cache-dir',required=True);q.add_argument('--instance-id',required=True)
        q.add_argument('--selection',choices=['pilot','test','train40','application','final'],default='pilot')
        q.add_argument('--application-data-dir')
        q.add_argument('--workers',type=int,default=8);q.add_argument('--wall-limit-s',type=float,default=3600)
        if command=='evaluate':
            q.add_argument('--calibration',required=True);q.add_argument('--models')
            q.add_argument('--methods');q.add_argument('--runtime-freeze');q.add_argument('--device',default='cpu')
            q.add_argument('--resume-campaign',action='store_true')
            q.add_argument('--seeds',help='pilot only, e.g. 2026; formal requires all three')
    args=p.parse_args()
    if args.command=='worker':
        raise SystemExit(worker(args.request))
    if args.selection in ('application','final') and not args.application_data_dir:
        p.error('--application-data-dir is required for known competition application')
    doc=load_frozen_protocol(args.protocol)
    print(json.dumps(calibrate(args,doc) if args.command=='calibrate' else evaluate_campaign(args,doc)))


if __name__=='__main__':
    main()
