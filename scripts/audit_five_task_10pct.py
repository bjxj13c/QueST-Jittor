"""Check five-task goal isolation,or audit actual training and held-out results."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import h5py

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/five_task_10pct_20261029';DATA=RUN/'dataset'
def read(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def key(value):return value.astype(np.float32).astype(np.float64).round(7).tobytes()

def isolation(p):
    os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    seen={t:set() for t in p['tasks']}
    roots=[ROOT,ROOT/'outputs/official_reach_100_20261028/adapted',
        ROOT/'outputs/overnight_20261029/datasets/reach280',ROOT/'outputs/overnight_20261029/datasets/multi240',DATA]
    for root in roots:
        m=read(root/'records/reach_split_v1.json');names=m.get('task_names',['reach-v2']);ids=m.get('episode_task_ids',{})
        with h5py.File(root/m['dataset'],'r') as f:
            for ep in m['train']+m['validation']:
                task=names[ids.get(ep,0)]
                if task in seen:seen[task].add(key(f[ep]['observations'][0][-3:]))
    records=[]
    for task in p['tasks']:
        overlaps=[];unique=set()
        for seed in range(p['test']['seed_start'],p['test']['seed_start']+p['test']['episodes_per_task']):
            env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+'-goal-observable'](seed=seed)
            try:
                obs,_=env.reset();g=key(obs[-3:]);unique.add(g)
                if g in seen[task]:overlaps.append(seed)
            finally:env.close()
        records.append({'task':task,'known_goals':len(seen[task]),'test_unique_goals':len(unique),'overlap_seeds':overlaps})
    report={'scope':'goal-only audit before any learned-policy test;includes old tokenizer data','records':records}
    (RUN/'test_isolation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert all(not r['overlap_seeds'] for r in records),'Preserve conflict and register new seeds before testing'

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]

def report(p):
    out=RUN/'experiments';m=read(DATA/'records/reach_split_v1.json');assert len(m['train'])==450 and len(m['validation'])==100
    assert digest(DATA/m['dataset'])==m['sha256'];assert not set(m['train'])&set(m['validation']);audit={}
    with h5py.File(DATA/m['dataset'],'r') as f:
        for i,task in enumerate(p['tasks']):
            a=[e for e in m['train'] if m['episode_task_ids'][e]==i];b=[e for e in m['validation'] if m['episode_task_ids'][e]==i]
            assert len(a)==90 and len(b)==20 and not {key(f[e]['observations'][0][-3:]) for e in a}&{key(f[e]['observations'][0][-3:]) for e in b}
        for split in ('train','validation'):
            ids=np.concatenate([np.full(len(f[e]['actions']),int(e.split('_')[-1]),dtype=np.int32) for e in m[split]])
            starts=np.concatenate([np.arange(len(f[e]['actions']),dtype=np.int32) for e in m[split]])
            with np.load(DATA/'assets/canonical_labels'/f'{split}.npz') as labels:
                np.testing.assert_array_equal(labels['episode_id'],ids);np.testing.assert_array_equal(labels['start'],starts)
                assert labels['labels'].shape==(len(ids),8)
    audit['450_actual_train_and100_heldout_validation_with_canonical_order']=True
    for framework in ('torch','jittor'):
        token=DATA/'outputs'/('five_tokenizer_'+framework);s=read(token/'summary.json');assert s['status']=='complete'
        c=read(token/'config.json');assert c['dataset_sha256']==m['sha256']
        assert c['batch_plan_sha256']==digest(out/'tokenizer_batch_plan.npz')
        if framework=='jittor':assert not s['torch_imported']
        if framework=='jittor':
            with np.load(token/'batch_orders.npz') as a,np.load(out/'tokenizer_batch_plan.npz') as b:
                for name in a.files:np.testing.assert_array_equal(a[name],b[name])
            assert s['final']['epoch']==10
        else:assert s['epochs_completed']==10 and not s['partial_epoch']
    audit['both_tokenizers_used450train_common10epoch_plan']=True
    freeze=read(out/'test_freeze.json');results={};streams=[]
    for framework in ('torch','jittor'):
        folder=out/('policy_'+framework);s=read(folder/'summary.json');c=read(folder/'config.json')
        assert s['status']=='completed' and s['updates']==p['policy']['updates'] and s['seen_examples']==p['policy']['updates']*16
        assert c['dataset_sha256']==m['sha256'] and c['config']['tasks']==5 and c['train_episodes']==m['train']
        assert c['initial_policy_npz_sha256']==digest(out/'initial_five.npz')
        assert c['tokenizer_weights_sha256']==digest(DATA/'assets/weights.npz')
        assert digest(folder/'best_weights.npz')==freeze['models'][framework]
        gradients=[json.loads(line) for line in (folder/'gradients.jsonl').read_text().splitlines()]
        assert [g['step'] for g in gradients]==list(range(1,p['policy']['updates']+1))
        norms=np.array([g['preclip_norm'] for g in gradients]);assert np.isfinite(norms).all() and not s['nonfinite_loss_or_gradient']
        if framework=='jittor':assert not s['torch_imported']
        streams.append(s['batch_stream_sha256']);tasks={}
        with np.load(folder/'best_weights.npz') as w:assert w['task_encoder.weight'].shape==(5,384)
        for i,task in enumerate(p['tasks']):
            test=read(DATA/'outputs'/f'test_{framework}_{task}'/'summary.json')
            assert test['environment']==task+'-goal-observable' and test['task_id']==i and test['seed_set']=='experiment'
            assert [r['seed'] for r in test['episodes']]==list(range(p['test']['seed_start'],p['test']['seed_start']+p['test']['episodes_per_task']))
            if framework=='jittor':assert not test['torch_imported']
            successes=sum(bool(row['success']) for row in test['episodes']);n=len(test['episodes'])
            tasks[task]={'successes':successes,'episodes':n,'rate':successes/n,'wilson95':wilson(successes,n)}
        results[framework]={'training':s,'tasks':tasks,'max_grad_norm':float(norms.max())}
    assert len(set(streams))==1;audit['paired_policy12000_updates_and300_real_tests']=True
    with np.load(out/'fixture_torch.npz') as a,np.load(out/'fixture_jittor.npz') as b:
        for name in a.files:np.testing.assert_allclose(a[name],b[name],atol=2e-5,rtol=2e-4)
        np.testing.assert_array_equal(a['tokens'],b['tokens'])
    audit['same_learned_weights_alignment']=True
    assert all(not row['overlap_seeds'] for row in read(RUN/'test_isolation.json')['records']);audit['new_goal_isolation']=True
    result={'status':'audited','audit':audit,'results':results,'data':read(DATA/'data_summary.json'),
        'limitations':['one training seed','five tasks,not45','old native tokenizer initialization,notfromscratch','window-weighted task sampling','all failed demos retained','no post-test tuning','not a controlled causal comparison to previous3task settings']}
    dest=ROOT/'reports/FIVE_TASK_10PCT_RESULTS_20261029';dest.with_suffix('.json').write_text(json.dumps(result,indent=2))
    lines=['# 五任务、450条训练示范实验','',
        '实际训练示范450条，占作者4500条预训练示范的10%；另100条仅作验证。两个框架均完成分词器适配及共享五任务策略训练。','',
        '| 任务 | PyTorch成功/30 | Jittor成功/30 |','|---|---:|---:|']
    for t in p['tasks']:lines.append(f"| {t} | {results['torch']['tasks'][t]['successes']}/30 | {results['jittor']['tasks'][t]['successes']}/30 |")
    lines+=['','每个框架一个训练种子；按验证CE选择检查点后冻结，新测试种子12000–12029。分词器从旧原生权重适配10轮，两端使用共同初始化与批次计划；条件策略均使用冻结的原生分词器。',
        '各任务90条训练、20条验证，按目标位置隔离；所有失败专家示范保留，详情见data_summary.json。策略只输入RGB、8维机器人状态和任务ID。',
        '本轮同时改变任务数、分词器和训练预算，不能仅凭与旧实验的成绩差异断言增加数据必然有效；10%是轨迹数比例，不是论文复现完成度。','',
        '完整日志、配置、逐步梯度、权重及逐场景动作位于 outputs/five_task_10pct_20261029。']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for j,framework in enumerate(('torch','jittor')):
        values=[results[framework]['tasks'][t]['rate'] for t in p['tasks']]
        axes[0].bar(np.arange(5)+(j-.5)*.36,values,width=.36,label=framework)
        rows=[json.loads(line) for line in (out/('policy_'+framework)/'metrics.jsonl').read_text().splitlines()]
        axes[1].plot([r['step'] for r in rows],[r['validation_ce'] for r in rows],label=framework)
    axes[0].set_xticks(range(5),['Reach','Reach-wall','Push','Door-open','Window-open'],rotation=25)
    axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Task success rate;30scenes/task')
    axes[1].set_xlabel('Updates');axes[1].set_ylabel('Held-out validation CE')
    for ax in axes:ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('450 actual training demonstrations;one training seed/framework');fig.tight_layout();fig.savefig(dest.with_suffix('.png'),dpi=170);plt.close(fig)
    print(json.dumps({'audit':audit,'tasks':{f:results[f]['tasks'] for f in results}},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['isolation','report'],required=True);args=parser.parse_args()
    protocol=read(ROOT/'records/five_task_10pct_20261029.json')
    (isolation if args.phase=='isolation' else report)(protocol)
