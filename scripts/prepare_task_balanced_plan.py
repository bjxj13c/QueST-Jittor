"""CPU-only balanced training-window schedule;keepsall450train episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.data import ActionChunkDataset,load_manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--updates',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=320500);args=parser.parse_args()
    assert args.updates>0 and not args.output.exists()
    m=load_manifest(args.root,verify_hash=True);dataset=ActionChunkDataset(args.root,'train')
    task_ids=np.array([dataset.task_ids.get(ep,0) for ep,_ in dataset.index],dtype=np.int64)
    tasks=m['task_names'];n=len(tasks);assert n==5 and len(m['train'])==450
    pools=[np.flatnonzero(task_ids==i) for i in range(n)];assert all(len(v)>0 for v in pools)
    rng=np.random.default_rng(args.seed);orders=[rng.permutation(v) for v in pools];cursors=[0]*n
    plan=np.empty((args.updates,16),dtype=np.int64)
    for step in range(args.updates):
        batch_tasks=(np.arange(16)+16*step)%n;rng.shuffle(batch_tasks)
        for task in range(n):
            slots=np.flatnonzero(batch_tasks==task);picked=[];needed=len(slots)
            while needed:
                count=min(needed,len(orders[task])-cursors[task]);picked.extend(orders[task][cursors[task]:cursors[task]+count])
                cursors[task]+=count;needed-=count
                if cursors[task]==len(orders[task]):orders[task]=rng.permutation(pools[task]);cursors[task]=0
            plan[step,slots]=picked
    counts=np.bincount(task_ids[plan].ravel(),minlength=n);assert counts.max()-counts.min()<=1
    visits=np.bincount(plan.ravel(),minlength=len(dataset));assert visits.min()>0
    metadata={'scope':'uniformtaskmixture,windows shuffled within each task;no filtering or relabeling',
        'dataset_sha256':m['sha256'],'train_episodes':m['train'],'task_names':tasks,'updates':args.updates,'batch_size':16,'seed':args.seed,
        'task_training_windows':{t:len(pools[i]) for i,t in enumerate(tasks)},
        'baseline_window_mixture':{t:len(pools[i])/len(dataset) for i,t in enumerate(tasks)},
        'planned_task_exposures':{t:int(counts[i]) for i,t in enumerate(tasks)},
        'planned_task_mixture':{t:float(counts[i]/counts.sum()) for i,t in enumerate(tasks)},
        'all_train_windows_covered':True,'minimum_window_visits':int(visits.min()),
        'batch_stream_sha256':hashlib.sha256(plan.tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,indices=plan,metadata=np.asarray(json.dumps(metadata)))
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k!='train_episodes'},indent=2),flush=True)

if __name__=='__main__':main()
