"""Task-balanced fixed-update experiment;adapted from unchanged train_controlled_policy.py."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import PolicyConfig,TokenizerConfig
from quest_repro.data import load_manifest

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def augment(images,rng,pad):
    if not pad:return images
    padded=np.pad(images,((0,0),(0,0),(0,0),(pad,pad),(pad,pad)),mode='edge')
    result=np.empty_like(images)
    for i,(y,x) in enumerate(rng.integers(0,2*pad+1,(len(images),2))):result[i]=padded[i,:,:,y:y+128,x:x+128]
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--framework',choices=['torch','jittor'],required=True)
    parser.add_argument('--tokenizer-weights',type=Path,required=True);parser.add_argument('--tokenizer-config',type=Path,required=True)
    parser.add_argument('--tokenizer-checkpoint',type=Path);parser.add_argument('--initial-weights',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=4000);parser.add_argument('--eval-every',type=int,default=400)
    parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--dropout',type=float,default=0);parser.add_argument('--shift',type=int,default=0)
    parser.add_argument('--seed',type=int,default=305000)
    parser.add_argument('--canonical-labels',type=Path)
    parser.add_argument('--batch-plan',type=Path,required=True)
    args=parser.parse_args()
    if args.updates<1 or args.eval_every<1 or not 0<=args.shift<=8:raise ValueError('Invalid update/eval/shift settings')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=load_manifest(args.root,verify_hash=True)
    cfg=PolicyConfig(tasks=len(manifest.get('task_names',['reach-v2'])),dropout=args.dropout)
    token_cfg=TokenizerConfig(**json.loads(args.tokenizer_config.read_text())['config'])
    if args.framework=='torch':
        import torch
        from torch.utils.data import DataLoader
        from quest_repro.policy_torch import QueSTPolicy
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state,optimizer_groups,seed_all
        from quest_repro.policy_data import PolicyDataset
        from train_policy import evaluate
        seed_all(args.seed);torch.set_num_threads(4)
        model=QueSTPolicy(cfg).cuda();load_npz_state(model,args.initial_weights)
        tokenizer=SkillVAE(token_cfg).cuda().eval();load_npz_state(tokenizer,args.tokenizer_weights)
        opt=torch.optim.AdamW(optimizer_groups(model,args.weight_decay),lr=args.lr,eps=1e-8,betas=(.9,.999))
        datasets={s:PolicyDataset(args.root,s,tokenizer,args.tokenizer_weights,batch_size=16,device='cuda') for s in ('train','validation')}
        loader=DataLoader(datasets['validation'],batch_size=16,shuffle=False,generator=torch.Generator().manual_seed(0))
        def evaluate_current():
            m=evaluate(model,tokenizer,loader,torch.device('cuda'))
            return {'validation_ce':m['teacher_forced_ce'],'autoregressive_valid_mae':m['autoregressive_action_mae_valid'],'autoregressive_accuracy':m['autoregressive_token_accuracy']}
        def save_weights(name):
            np.savez_compressed(args.output/f'{name}_weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg.to_dict(),
                'extra':{**metadata,'tokenizer_sha256':sha(args.tokenizer_checkpoint)},'step':step},args.output/f'{name}.pt')
    else:
        import jittor as jt
        from quest_repro.policy_jittor import QueSTPolicy
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        from quest_repro.policy_data_jittor import PolicyDataset
        from quest_repro.jittor_training import optimizer_groups,save_npz
        from train_policy_jittor import evaluate
        assert 'torch' not in sys.modules
        jt.set_global_seed(args.seed);jt.flags.use_cuda=1
        model=QueSTPolicy(cfg);strict_load_npz(model,args.initial_weights)
        tokenizer=SkillVAE(token_cfg);strict_load_npz(tokenizer,args.tokenizer_weights);tokenizer.eval()
        opt=jt.optim.AdamW(optimizer_groups(model,args.weight_decay),lr=args.lr,eps=1e-8,betas=(.9,.999))
        datasets={s:PolicyDataset(args.root,s,tokenizer,args.tokenizer_weights,batch_size=16) for s in ('train','validation')}
        def evaluate_current():
            m=evaluate(model,tokenizer,datasets['validation'],16)
            return {key:m[key] for key in ('validation_ce','autoregressive_valid_mae','autoregressive_accuracy')}
        def save_weights(name):save_npz(model,args.output/f'{name}_weights.npz')
    with np.load(args.batch_plan,allow_pickle=False) as plan_archive:
        sample_plan=plan_archive['indices'].copy()
        sample_info=json.loads(str(plan_archive['metadata']))
    assert sample_plan.ndim==2 and sample_plan.shape[1]==16 and len(sample_plan)>=args.updates
    assert sample_plan.dtype==np.int64 and sample_plan.min()>=0 and sample_plan.max()<len(datasets['train'])
    assert sample_info['dataset_sha256']==manifest['sha256']
    assert sample_info['train_episodes']==manifest['train']
    label_audit={} 
    if args.canonical_labels:
        for split,dataset in datasets.items():
            path=args.canonical_labels/f'{split}.npz'
            with np.load(path) as archive:canonical=archive['labels'].copy()
            if canonical.shape!=dataset.labels.shape:raise ValueError('Canonical label shape mismatch')
            label_audit[split]={'mismatches_before_using_canonical':int(np.sum(dataset.labels!=canonical)),
                'token_count':int(canonical.size),'canonical_sha256':sha(path)}
            dataset.labels=canonical.astype(np.int64 if args.framework=='torch' else np.int32)
    metadata={'framework':args.framework,'config':cfg.to_dict(),'tokenizer_weights_sha256':sha(args.tokenizer_weights),
        'dataset_sha256':manifest['sha256'],'train_episodes':manifest['train'],'validation_episodes':manifest['validation'],
        'initial_policy_npz_sha256':sha(args.initial_weights),'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'scope':'fixed-update controlled policy training;not policy success','batch_size':16,'sampling':'continuous shuffled epochs,exact16samples/update;no padded minibatch',
        'normalization':'RGB/255 exactly once;no privileged policy state'}
    metadata['label_audit']=label_audit
    metadata['sampling_plan_sha256']=sha(args.batch_plan)
    metadata['sampling_plan_metadata']=sample_info
    metadata['reference_trainer_sha256']=sha(ROOT/'scripts/train_controlled_policy.py')
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2))
    initial=evaluate_current();(args.output/'initial_validation.json').write_text(json.dumps(initial,indent=2))
    aug_rng=np.random.default_rng(args.seed+1)
    plan_hash=hashlib.sha256()
    best=float('inf');best_step=0;norms=[];loss_sum=0.;interval_steps=0;started=time.monotonic();step=0
    columns=['step','train_ce','validation_ce','autoregressive_valid_mae','autoregressive_accuracy','lr','grad_norm_max_interval','elapsed_seconds']
    with (args.output/'metrics.csv').open('w',newline='') as file,(args.output/'gradients.jsonl').open('w') as gradfile:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        for step in range(1,args.updates+1):
            ids=sample_plan[step-1]
            plan_hash.update(ids.tobytes())
            items=[datasets['train'][int(i)] for i in ids]
            batch={key:np.stack([item[key] for item in items]) for key in ('images','robot_states','task_id','targets')}
            batch['images']=augment(batch['images'],aug_rng,args.shift)
            lr=1e-5+(args.lr-1e-5)*(1+math.cos(math.pi*(step-1)/args.updates))/2
            model.train()
            if args.framework=='torch':
                for group in opt.param_groups:group['lr']=lr
                images,state,task,targets=[torch.from_numpy(batch[k]).cuda() for k in ('images','robot_states','task_id','targets')]
                opt.zero_grad(set_to_none=True);_,loss=model(images,state,task.long(),targets.long());loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),100.).item());value=float(loss.item())
                if not np.isfinite(norm) or not np.isfinite(value):raise FloatingPointError('Nonfinite loss/gradient')
                opt.step()
            else:
                opt.lr=lr
                images,state,task,targets=[jt.array(batch[k]) for k in ('images','robot_states','task_id','targets')]
                _,loss=model(images,state,task.int32(),targets.int32());value=float(loss.numpy())
                opt.zero_grad();opt.backward(loss)
                squared=[(g*g).sum() for group in opt.param_groups for p,g in zip(group['params'],group['grads']) if not p.is_stop_grad()]
                norm=float(jt.sqrt(sum(squared)).numpy())
                if not np.isfinite(norm) or not np.isfinite(value):raise FloatingPointError('Nonfinite native loss/gradient')
                opt.clip_grad_norm(100.);opt.step()
            norms.append(norm);loss_sum+=value;interval_steps+=1
            gradfile.write(json.dumps({'step':step,'preclip_norm':norm,'clipped':norm>100})+'\n')
            if step%100==0:print(json.dumps({'phase':'training','step':step,'loss':value,'grad_norm':norm}),flush=True);gradfile.flush()
            if step%args.eval_every==0 or step==args.updates:
                metrics=evaluate_current()
                row={'step':step,'train_ce':loss_sum/interval_steps,**metrics,'lr':lr,'grad_norm_max_interval':max(norms[-interval_steps:]),'elapsed_seconds':time.monotonic()-started}
                writer.writerow(row);file.flush();print(json.dumps(row),flush=True)
                with (args.output/'metrics.jsonl').open('a') as logs:logs.write(json.dumps(row)+'\n')
                if metrics['validation_ce']<best:
                    best=metrics['validation_ce'];best_step=step;save_weights('best')
                    (args.output/'best_validation.json').write_text(json.dumps({'step':step,**metrics},indent=2))
                save_weights('last');loss_sum=0.;interval_steps=0
    result={'status':'completed','framework':args.framework,'updates':step,'seen_examples':step*16,'best_validation_ce':best,'best_step':best_step,
        'batch_stream_sha256':plan_hash.hexdigest(),'preclip_gradient_norm_max':max(norms),'preclip_gradient_norm_p95':float(np.percentile(norms,95)),
        'gradient_clip_fraction':float(np.mean(np.asarray(norms)>100)),'nonfinite_loss_or_gradient':False,
        'elapsed_seconds':time.monotonic()-started,'torch_imported':'torch' in sys.modules}
    if args.framework=='jittor':assert not result['torch_imported']
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
