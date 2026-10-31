"""Activation grid tokenizer training;all-position L1;fresh common initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from quest_repro.config import TokenizerConfig
from quest_repro.data import ActionChunkDataset,load_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--framework',choices=['torch','jittor'],required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--masked',action='store_true');p.add_argument('--updates',type=int,default=22180);p.add_argument('--eval-every',type=int,default=2218);p.add_argument('--seed',type=int,default=520500);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import os
    assert os.environ.get('QUEST_ACTIVATION_CONV') and os.environ.get('QUEST_ACTIVATION_FFN'), 'Use run_activation_variant.py'
    assert not a.masked, 'Activation grid keeps original all-position L1'
    m=load_manifest(a.root,verify_hash=True);cfg=TokenizerConfig(**json.loads(a.config.read_text())['config']);assert cfg.attn_pdrop==0
    datasets={s:ActionChunkDataset(a.root,s) for s in ('train','validation')};rng=np.random.default_rng(a.seed)
    if a.framework=='torch':
        import torch
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state,optimizer_groups
        torch.manual_seed(a.seed);torch.set_num_threads(4);model=SkillVAE(cfg).cuda();load_npz_state(model,a.initial)
        opt=torch.optim.AdamW(optimizer_groups(model,1e-4),lr=1e-4,betas=(.9,.999),eps=1e-8)
        def prediction(x):return model(torch.from_numpy(x).cuda())[0]
        def as_numpy(x):return x.detach().cpu().numpy()
        def save():np.savez_compressed(a.output/'best_weights.npz',**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
    else:
        import jittor as jt
        from quest_repro.tokenizer_jittor import SkillVAE,strict_load_npz
        from quest_repro.jittor_training import optimizer_groups,save_npz
        assert 'torch' not in sys.modules;jt.set_global_seed(a.seed);jt.flags.use_cuda=1;model=SkillVAE(cfg);strict_load_npz(model,a.initial)
        opt=jt.optim.AdamW(optimizer_groups(model,1e-4),lr=1e-4,betas=(.9,.999),eps=1e-8)
        def prediction(x):return model.forward_details(jt.array(x))['prediction']
        def as_numpy(x):return x.numpy()
        def save():save_npz(model,a.output/'best_weights.npz')
    def evaluate():
        model.eval();error=valid=total=positions=0.
        context=torch.no_grad() if a.framework=='torch' else jt.no_grad()
        with context:
            for start in range(0,len(datasets['validation']),32):
                items=[datasets['validation'][i] for i in range(start,min(start+32,len(datasets['validation'])))];x=np.stack([v['actions'] for v in items]);mask=np.stack([v['pad_mask'] for v in items]);e=np.abs(as_numpy(prediction(x))-x)
                error+=float(e.sum());total+=e.size;valid+=float((e*mask).sum());positions+=float(np.broadcast_to(mask,e.shape).sum())
        return {'mae_all':error/total,'mae_valid':valid/positions}
    metadata={'config':cfg.to_dict(),'dataset_sha256':m['sha256'],'initial_sha256':hashlib.sha256(a.initial.read_bytes()).hexdigest(),'loss':'valid-only masked L1' if a.masked else 'all-position L1 including repeated padding','selection':'validation all-position MAE','framework':a.framework,'updates':a.updates,'conv_activation':os.environ['QUEST_ACTIVATION_CONV'],'ffn_activation':os.environ['QUEST_ACTIVATION_FFN'],'seed':a.seed,'eval_every':a.eval_every}
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2));initial=evaluate();(a.output/'initial_validation.json').write_text(json.dumps(initial))
    order=rng.permutation(len(datasets['train']));cursor=0;stream=hashlib.sha256();best=float('inf');started=time.monotonic();norms=[]
    with (a.output/'gradients.jsonl').open('w') as log:
        for step in range(1,a.updates+1):
            ids=[]
            while len(ids)<16:
                n=min(16-len(ids),len(order)-cursor);ids.extend(order[cursor:cursor+n]);cursor+=n
                if cursor==len(order):order=rng.permutation(len(order));cursor=0
            stream.update(np.asarray(ids,dtype=np.int64).tobytes());items=[datasets['train'][i] for i in ids];x=np.stack([v['actions'] for v in items]);mask=np.stack([v['pad_mask'] for v in items]).astype(np.float32);model.train()
            lr=1e-5+(1e-4-1e-5)*(1+np.cos(np.pi*(step-1)/a.updates))/2
            if a.framework=='torch':
                for group in opt.param_groups:group['lr']=float(lr)
                opt.zero_grad(set_to_none=True);e=(prediction(x)-torch.from_numpy(x).cuda()).abs()
                loss=(e*torch.from_numpy(mask).cuda()).sum()/(float(mask.sum())*4) if a.masked else e.mean();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),100));value=float(loss.detach());
                if not np.isfinite(value) or not np.isfinite(norm):raise FloatingPointError('Nonfinite loss/gradient before update')
                opt.step()
            else:
                opt.lr=float(lr);e=(prediction(x)-jt.array(x)).abs();loss=(e*jt.array(mask)).sum()/(float(mask.sum())*4) if a.masked else e.mean();value=float(np.asarray(loss.numpy()).reshape(-1)[0]);opt.zero_grad();opt.backward(loss)
                normsquared=[(g*g).sum() for group in opt.param_groups for v,g in zip(group['params'],group['grads']) if not v.is_stop_grad()];norm=float(np.asarray(jt.sqrt(sum(normsquared)).numpy()).reshape(-1)[0]);
                if not np.isfinite(value) or not np.isfinite(norm):raise FloatingPointError('Nonfinite native loss/gradient before update')
                # 检查通过后裁剪梯度并更新分词器；裁剪前范数另存日志。
                opt.clip_grad_norm(100);opt.step()
            if not np.isfinite(value) or not np.isfinite(norm):raise FloatingPointError('Nonfinite tokenizer loss/gradient')
            norms.append(norm);log.write(json.dumps({'step':step,'loss':value,'gradient_norm':norm})+'\n')
            if step%100==0:log.flush();print(json.dumps({'step':step,'loss':value}),flush=True)
            if step%a.eval_every==0 or step==a.updates:
                scores=evaluate();print(json.dumps({'step':step,**scores}),flush=True)
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**scores})+'\n')
                # 按验证集重建MAE选择检查点；mae_all包含补齐位置。
                if scores['mae_all']<best:best=scores['mae_all'];save();(a.output/'best_validation.json').write_text(json.dumps({'step':step,**scores}))
    result={'status':'completed','updates':a.updates,'best_validation_mae':best,'batch_stream_sha256':stream.hexdigest(),'max_gradient_norm':max(norms),'finite':True,'torch_imported':'torch' in sys.modules,'elapsed_seconds':time.monotonic()-started}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
