"""Bounded train-only counterfactual collection with all attempts retained."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import time

from .runtime import canonical_hash, environment, load_scene, source_hash, write_new
from .protocol import load_frozen_protocol


def run(args):
    from .counterfactual import collect_state
    from .features import FeatureContext
    from .plugin import initialize
    from .search import search, schedule_hash
    from .inference import Ranker
    document = load_frozen_protocol(args.protocol)
    if not 1 <= args.max_evaluations <= 5120 or not 1 <= args.max_states <= 8:
        raise ValueError('collection limits exceed the registered first batch')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();counts=Counter();samples=0;failures=[]
    env=environment(args.instance_id)
    write_new(out/'environment.json',env)
    write_new(out/'config.json',vars(args))
    ranker=None
    code=source_hash();dependencies=canonical_hash(env['versions'])
    for cid in document['splits']['train_subsets']['40']:
        if samples >= args.max_evaluations or time.perf_counter()-started >= args.max_wall_s:
            break
        try:
            pb,meta=load_scene(args.dataset_root,cid,args.cache_dir,allowed_splits=('train',))
            initial,mu,info=initialize(pb,{'initial_h_iters':300,'initial_h_fast':False,
                                         'workers':args.workers,'seed':2026})
            write_new(out/cid/'initial.json',{'metadata':meta,'info':info})
            context=FeatureContext(pb,mu)
            if args.checkpoint:
                expected={k:context.fingerprint[k] for k in ('feature_version','pool_version','price_version')}
                ranker=Ranker(args.checkpoint,device=args.device,expected_versions=expected,
                              allow_static_domain_transfer=True)
            learned_mode='neural' if ranker is not None and ranker.model.dynamic_dim else 'static_only'
            state_modes=['rule','random','rule','random','rule',learned_mode,'rule',learned_mode] if ranker else ['rule','random']*4
            seen_parents=set()
            for index,mode in enumerate(state_modes[:args.max_states]):
                left=args.max_evaluations-samples
                if left<=0 or time.perf_counter()-started >= args.max_wall_s:
                    break
                # Each sampler starts from the fixed baseline. The initial
                # state is itself retained when a sampler finds no improvement.
                parent,trajectory=search(pb,initial,deadline=min(started+args.max_wall_s,time.perf_counter()+2.),
                     mode=mode,seed=2026+index,max_rounds=index+1,ranker=ranker,feature_context=context)
                write_new(out/cid/f'trajectory-{index}.json',trajectory)
                parent_hash=schedule_hash(parent)
                if parent_hash in seen_parents:
                    counts['duplicate_parent_skipped']+=1
                    continue
                seen_parents.add(parent_hash)
                rows=collect_state(pb,parent,context,catalog_id=cid,scene_id=meta['scene_id'],
                         split='train',initial_plan_hash=schedule_hash(initial),seed=2026,
                         round_index=index,max_candidates=min(16,left),max_trials=64,timeout_s=2.,
                         code_fingerprint=code,dependency_fingerprint=dependencies)
                with (out/'samples.jsonl').open('a') as stream:
                    for row in rows:
                        row['state_sampler_mode']=mode
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        counts[row['status']]+=1;samples+=1
        except Exception as exc:
            failure={'catalog_id':cid,'status':'failed','error':type(exc).__name__+': '+str(exc)}
            failures.append(failure)
            with (out/'failures.jsonl').open('a') as stream:
                stream.write(json.dumps(failure)+'\n')
    elapsed=time.perf_counter()-started
    summary={'evaluations':samples,'statuses':dict(counts),'failures':failures,'wall_s':elapsed,
             'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'rss_unit':'bytes on macOS, KiB on Linux; subprocess peaks are separately recorded',
             'operations_per_s':samples/elapsed,'requested_calibration_operations':100,
             'calibration_complete':samples>=100,'protocol_freeze':document['freeze']['freeze_hash'],
             'code_hash':code,'quality_claim':False}
    write_new(out/'summary.json',summary)
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache-dir',required=True);p.add_argument('--instance-id',required=True)
    p.add_argument('--protocol',default='experiments/neural-lns-v1');p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-evaluations',type=int,default=100);p.add_argument('--max-states',type=int,default=8)
    p.add_argument('--max-wall-s',type=float,default=3600);p.add_argument('--checkpoint');p.add_argument('--device',default='cpu')
    args=p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        out=Path(args.output)
        if out.exists():
            with (out/'attempts.jsonl').open('a') as stream:
                stream.write(json.dumps({'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                    'error':type(exc).__name__+': '+str(exc),'samples_retained':(out/'samples.jsonl').exists()})+'\n')
        raise


if __name__=='__main__':main()
