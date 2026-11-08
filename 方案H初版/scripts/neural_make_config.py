"""Create a portable frozen-subset training config without running training."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset-root',required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--subset',type=int,choices=(40,100,200,420),default=40)
p.add_argument('--seed',type=int,choices=(2026,2027,2028),default=2026)
p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
p.add_argument('--samples');p.add_argument('--init-checkpoint')
args=p.parse_args()
config={'dataset_root':args.dataset_root,'splits_path':'experiments/neural-lns-v1/splits.json',
        'schema_path':'experiments/neural-lns-v1/feature_schema.json','subset':args.subset,
        'seed':args.seed,'steps':10000,'batch_size':8192,'learning_rate':1e-3,
        'weight_decay':1e-4,'eval_every':500,'checkpoint_every':500,
        'device':args.device,'threads':1,'batches_per_scene':16}
if args.samples:config['samples_path']=args.samples
if args.init_checkpoint:config['init_checkpoint']=args.init_checkpoint
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as f:json.dump(config,f,indent=2);f.write('\n')
print(args.output)
