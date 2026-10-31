"""Inspect actual factorial/native/multitask evidence before generating claims."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import h5py

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();night=args.root/'outputs/overnight_20261029';out=night/'experiments'
workflow=json.loads((out/'workflow.json').read_text());assert workflow['status']=='completed'
extension=json.loads((out/'native_factorial_workflow.json').read_text());assert extension['status']=='completed'
protocol=workflow['protocol'];audit={};records={};native=workflow['selected_single_case_by_validation']
def read(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]
def inspect_training(name):
    folder=out/name;s=read(folder/'summary.json');c=read(folder/'config.json')
    assert s['status']=='completed' and s['updates']==4000 and s['seen_examples']==64000
    gradients=[json.loads(line) for line in (folder/'gradients.jsonl').read_text().splitlines()]
    assert [g['step'] for g in gradients]==list(range(1,4001))
    norms=np.array([g['preclip_norm'] for g in gradients]);assert np.isfinite(norms).all()
    assert float(norms.max())==s['preclip_gradient_norm_max'] and not s['nonfinite_loss_or_gradient']
    if c['framework']=='jittor':assert not s['torch_imported']
    return {'best_validation_ce':s['best_validation_ce'],'best_step':s['best_step'],'updates':4000,'seen_windows':64000,
        'seconds':s['elapsed_seconds'],'grad_norm_max':s['preclip_gradient_norm_max'],'grad_norm_p95':s['preclip_gradient_norm_p95'],
        'clip_fraction':s['gradient_clip_fraction'],'batch_stream_sha256':s['batch_stream_sha256'],
        'initial_sha256':c['initial_policy_npz_sha256'],'dataset_sha256':c['dataset_sha256'],'label_audit':c['label_audit']}
def inspect_eval(result,start,n):
    assert result['seed_set']=='experiment' and result['per_episode_sampling']
    assert [e['seed'] for e in result['episodes']]==list(range(start,start+n))
    assert result['top_k']==5 and result['horizon']==8 and result['max_steps']==150
    if result['framework']=='jittor':assert not result['torch_imported']
    y=np.array([bool(e['success']) for e in result['episodes']],dtype=np.int32);k=int(y.sum())
    return {'successes':k,'episodes':n,'rate':k/n,'wilson95':wilson(k,n),'outcomes':y.tolist()}
freeze=read(out/'single_test_freeze.json')
for name,settings in protocol['four_cases'].items():
    train=inspect_training(name);evaluation=inspect_eval(workflow['single_results'][name],8000,50)
    assert digest(out/name/'best_weights.npz')==freeze['candidates'][name]
    cfg=read(out/name/'config.json')['arguments']
    for key in ('lr','dropout','weight_decay','shift'):assert cfg[key]==settings[key]
    records[name]={**train,**evaluation,'settings':settings}
assert len({r['initial_sha256'] for r in records.values()})==1
assert native==min(records,key=lambda name:records[name]['best_validation_ce'])==freeze['selected_by_validation']
native_record={**inspect_training('selected_native'),**inspect_eval(workflow['single_results']['selected_native'],8000,50)}
assert native_record['batch_stream_sha256']==records[native]['batch_stream_sha256']
assert digest(out/'selected_native/best_weights.npz')==freeze['native_sha256']
audit['factorial_updates_initialization_and_gradients']=True;audit['validation_only_selection_and_checkpoint_identity']=True
native_matrix={};native_freeze=read(out/'native_factorial/test_freeze.json')
assert extension['registered_utc']<min(r['created_utc'] for r in workflow['single_results'].values())
for name,setting in protocol['four_cases'].items():
    subpath=Path('selected_native') if name==native else Path('native_factorial')/name
    result=extension['results'][name]
    row={**inspect_training(subpath),**inspect_eval(result['evaluation'],8000,50)}
    assert row['initial_sha256']==records[name]['initial_sha256']
    assert row['batch_stream_sha256']==records[name]['batch_stream_sha256']
    assert row['dataset_sha256']==records[name]['dataset_sha256']
    assert digest(out/subpath/'best_weights.npz')==native_freeze['cases'][name]
    cfg=read(out/subpath/'config.json')['arguments']
    for key in ('lr','dropout','weight_decay','shift'):assert cfg[key]==setting[key]
    native_matrix[name]=row
audit['full_four_case_native_replications']=True
small=read(night/'datasets/reach80/records/reach_split_v1.json');large=read(night/'datasets/reach280/records/reach_split_v1.json')
assert small['validation']==large['validation'] and set(small['train'])<set(large['train'])
assert len(small['train'])==80 and len(large['train'])==280
audit['nested_training_data_and_identical_validation']=True
for label in ['reach80','reach280','multi240']:
    root=night/'datasets'/label;m=read(root/'records/reach_split_v1.json')
    assert digest(root/m['dataset'])==m['sha256']
    assert read(root/'assets/canonical_labels/report.json')['dataset_sha256']==m['sha256']
    with h5py.File(root/m['dataset'],'r') as f:
        for split in ['train','validation']:
            ids=np.concatenate([np.full(len(f[name]['actions']),int(name.split('_')[-1]),dtype=np.int32) for name in m[split]])
            starts=np.concatenate([np.arange(len(f[name]['actions']),dtype=np.int32) for name in m[split]])
            with np.load(root/'assets/canonical_labels'/f'{split}.npz') as labels:
                np.testing.assert_array_equal(labels['episode_id'],ids);np.testing.assert_array_equal(labels['start'],starts)
                assert labels['labels'].shape==(len(ids),8) and labels['labels'].min()>=0 and labels['labels'].max()<1000
audit['actual_hdf5_hashes_and_canonical_target_order']=True
comparisons={}
for label in ['single','multi']:
    with np.load(out/f'{label}_fixture_torch.npz') as a,np.load(out/f'{label}_fixture_jittor.npz') as b:
        for k in a.files:np.testing.assert_allclose(a[k],b[k],atol=2e-5,rtol=2e-4)
        np.testing.assert_array_equal(a['tokens'],b['tokens'])
        comparisons[label]={'samples':len(a['tokens']),'tokens_exact':True,'tokens':int(a['tokens'].size),
            'context_max_abs':float(np.abs(a['context']-b['context']).max()),'actions_max_abs':float(np.abs(a['actions']-b['actions']).max())}
audit['same_learned_weights_port_alignment']=True
isolation=read(night/'test_isolation.json')
for group in isolation['datasets'].values():
    for result in group.values():assert not result['overlap_seeds']
audit['new_test_goal_isolation']=True
multi={}
for framework in ('torch','jittor'):
    train=inspect_training('multi_'+framework);cfg=read(out/('multi_'+framework)/'config.json')
    assert cfg['config']['tasks']==3
    with np.load(out/('multi_'+framework)/'best_weights.npz') as w:assert w['task_encoder.weight'].shape==(3,384)
    task_results={}
    for task_id,task in enumerate(protocol['multi_task_tasks']):
        result=workflow['multi_results'][framework+'_'+task]
        assert result['environment']==task+'-goal-observable' and result['task_id']==task_id
        task_results[task]=inspect_eval(result,9000,20)
    multi[framework]={'training':train,'tasks':task_results,'total_successes':sum(r['successes'] for r in task_results.values()),'total_episodes':60}
assert multi['torch']['training']['batch_stream_sha256']==multi['jittor']['training']['batch_stream_sha256']
token=read(night/'datasets/multi240/outputs/overnight_multi_tokenizer5/summary.json')
assert token['status']=='complete' and not token['torch_imported']
audit['genuine_shared_three_task_training_and_testing']=True
def paired_effect(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a);up=int(((a==1)&(b==0)).sum());down=int(((a==0)&(b==1)).sum());discord=up+down
    p=min(1.,2*sum(math.comb(discord,i) for i in range(min(up,down)+1))/2**discord) if discord else 1.
    rng=np.random.default_rng(306000);indices=rng.integers(0,n,(10000,n));delta=(a-b)[indices].mean(axis=1)
    return {'difference':float((a-b).mean()),'paired_bootstrap95':np.quantile(delta,[.025,.975]).tolist(),'improved_scenes':up,'worsened_scenes':down,'mcnemar_exact_p':p}
effects={name:paired_effect(records[name]['outcomes'],records['small_base']['outcomes']) for name in ['large_base','small_tuned','large_tuned']}
ordered=sorted(effects,key=lambda name:effects[name]['mcnemar_exact_p']);previous=0.
for i,name in enumerate(ordered):
    previous=max(previous,min(1.,effects[name]['mcnemar_exact_p']*(len(ordered)-i)));effects[name]['holm_adjusted_p']=previous
native_effects={name:paired_effect(native_matrix[name]['outcomes'],native_matrix['small_base']['outcomes']) for name in ['large_base','small_tuned','large_tuned']}
combined=[('torch',name,e) for name,e in effects.items()]+[('jittor',name,e) for name,e in native_effects.items()]
combined.sort(key=lambda item:item[2]['mcnemar_exact_p']);previous=0.
for i,(_,_,effect) in enumerate(combined):
    previous=max(previous,min(1.,effect['mcnemar_exact_p']*(6-i)));effect['holm_adjusted_p']=previous
interaction=np.asarray(records['large_tuned']['outcomes'])-np.asarray(records['large_base']['outcomes'])-np.asarray(records['small_tuned']['outcomes'])+np.asarray(records['small_base']['outcomes'])
indices=np.random.default_rng(306001).integers(0,50,(10000,50))
interaction_result={'difference_in_differences':float(interaction.mean()),'paired_bootstrap95':np.quantile(interaction[indices].mean(axis=1),[.025,.975]).tolist()}
native_interaction=np.asarray(native_matrix['large_tuned']['outcomes'])-np.asarray(native_matrix['large_base']['outcomes'])-np.asarray(native_matrix['small_tuned']['outcomes'])+np.asarray(native_matrix['small_base']['outcomes'])
native_interaction_result={'difference_in_differences':float(native_interaction.mean()),'paired_bootstrap95':np.quantile(native_interaction[indices].mean(axis=1),[.025,.975]).tolist()}
summary={'audit':audit,'factorial':records,'native_factorial':native_matrix,'selected_by_validation':native,'selected_native':native_record,'effects':effects,'native_effects':native_effects,
    'interaction':interaction_result,'native_interaction':native_interaction_result,'multitask':multi,'same_weights':comparisons,'data_quality':read(out/'data_quality.json'),
    'limitations':['one training seed per configuration','CIs reflect test-scene sampling,not training-seed variability','expanded data also broadens goal coverage','tuned settings are a bundle','3tasks and shortened training,not full45task paper']}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2))
names={'small_base':'80条＋原设置','large_base':'280条＋原设置','small_tuned':'80条＋调参组合','large_tuned':'280条＋调参组合'}
lines=['# 扩大数据、调参与三任务实验（2026-10-29）','',
    '本轮重新建立固定计算预算的2×2对照，不将旧55%成绩与不同测试集的成绩直接比较。下面所有结果来自实际训练和仿真，不保证每项改变都会提高成功率。','',
    '## 单任务四组对照','',
    '| 配置 | 成功/50 | 成功率 | 95% Wilson区间 | 最佳验证CE / 更新步 |',
    '|---|---:|---:|---:|---:|']
for name,r in records.items():lines.append(f"| {names[name]} | {r['successes']}/50 | {r['rate']:.0%} | {r['wilson95'][0]:.1%}–{r['wilson95'][1]:.1%} | {r['best_validation_ce']:.4f} / {r['best_step']} |")
lines+=['','训练集80条是280条的严格子集，验证集始终为同一20条。训练目标位置从40组增加到140组，因此这里检验的是“增加数据并扩大覆盖”，不是只增加同样轨迹的重复次数。每组固定4000次更新、batch16，共64000个窗口；同一初始权重、同一批次随机规则、相同验证频率，按验证CE选检查点。大数据没有获得额外更新预算。','',
    '原设置：学习率1e-4、dropout0、权重衰减1e-4。调参组合：学习率3e-5、dropout0.1、权重衰减1e-3、边缘填充后随机平移4像素。统一按更新次数余弦降至1e-5。调参同时改变多项设置，不能把效果单独归因于某个参数。','',
    '新测试种子为8000–8049，所有候选检查点在测试前冻结；top-5、温度1、执行8步、最多150步，各场景NumPy采样种子30000＋场景seed，避免前一场景执行长度改变后一场景随机数。旧1000–1019没有参与调参或本次测试。','',
    '### 相对80条原设置的配对比较','',
    '| 改动 | 成功率差 | 配对bootstrap 95%区间 | 改善/退步场景 | McNemar精确p（Holm校正） |','|---|---:|---:|---:|---:|']
for name,e in effects.items():lines.append(f"| {names[name]} | {e['difference']:+.0%} | {e['paired_bootstrap95'][0]:+.0%}–{e['paired_bootstrap95'][1]:+.0%} | {e['improved_scenes']}/{e['worsened_scenes']} | {e['holm_adjusted_p']:.4f} |")
lines+=['','上述Holm校正覆盖两框架共六项与基线的比较，均为探索性分析。','',
    '### Jittor四组独立重复','',
    '| 配置 | 成功/50 | 成功率 | 95% Wilson区间 | 相对Jittor基线 |','|---|---:|---:|---:|---:|']
for name,r in native_matrix.items():
    delta=r['rate']-native_matrix['small_base']['rate']
    lines.append(f"| {names[name]} | {r['successes']}/50 | {r['rate']:.0%} | {r['wilson95'][0]:.1%}–{r['wilson95'][1]:.1%} | {delta:+.0%} |")
lines+=['',f"PyTorch数据与调参的交互项（差分之差）为{interaction_result['difference_in_differences']:+.0%}，配对bootstrap区间{interaction_result['paired_bootstrap95'][0]:+.0%}–{interaction_result['paired_bootstrap95'][1]:+.0%}。Jittor交互项为{native_interaction_result['difference_in_differences']:+.0%}，区间{native_interaction_result['paired_bootstrap95'][0]:+.0%}–{native_interaction_result['paired_bootstrap95'][1]:+.0%}。这些是单次训练条件下的探索性结果；每种配置只训练一个随机种子，区间不包含重新训练造成的方差，不能据一次实验作普遍因果断言。",'',
    '## 原生Jittor重复验证','',
    f"测试前按验证CE选中的是“{names[native]}”。同设置、初始权重与批次流重新进行原生Jittor4000次更新，成功{native_record['successes']}/50（{native_record['rate']:.0%}），Torch为{records[native]['successes']}/50。原生训练/测试进程未导入Torch。dropout非零时，两框架随机掩码不要求相同；独立优化与同权重实现对齐分别报告。",'',
    '## 共享三任务策略','',
    '这部分使用一个共享策略和3个任务ID，不是三个单任务模型。Reach、Reach-wall、Push各100条示范，各80条训练、20条验证；先在混合训练动作上用Jittor适配分词器5轮，再冻结它分别训练Torch/Jittor策略4000次更新。此处改变了任务数、共享分词器和数据分布，属于更接近论文结构的可行性实验，不能只用它推断多任务的单独因果作用。','',
    '| 任务 | Torch成功/20 | Jittor成功/20 |','|---|---:|---:|']
for task in protocol['multi_task_tasks']:lines.append(f"| {task} | {multi['torch']['tasks'][task]['successes']}/20 | {multi['jittor']['tasks'][task]['successes']}/20 |")
lines+=['','三任务新测试种子为9000–9019，每个任务独立20场景。策略仅输入RGB、8维机器人状态及任务ID；39维完整状态不进入策略。Push专家只有90/100条成功，失败轨迹也按作者脚本保留；失败轨迹共5000/11006步，占约45%，会影响有效数据质量，不能把轨迹数直接等同于优质示范量。','',
    '## 梯度与实现核查','',
    '| 配置 | 最大裁剪前梯度范数 | P95 | 被裁剪比例 |','|---|---:|---:|---:|']
for name,r in list(records.items())+[(f'native_{name}',r) for name,r in native_matrix.items()]+[(f'multi_{name}',multi[name]['training']) for name in multi]:lines.append(f"| {names.get(name,name)} | {r['grad_norm_max']:.3f} | {r['grad_norm_p95']:.3f} | {r['clip_fraction']:.1%} |")
lines+=['','上述每个策略模型的4000次更新都有梯度记录并检查有限值，裁剪阈值为100；表中大范数和裁剪比例保留，不用“未报错”替代稳定性证据。分词器阶段沿用有限损失检查与裁剪，未额外记录逐步梯度范数。标签采用同一份原生分词器生成的静态目标；两框架独立计算标签与静态目标的差异先记录在config.label_audit中，再共享目标训练，未覆盖原缓存。','',
    f"同一组训练后权重的单任务比较{comparisons['single']['tokens']}个token、三任务比较{comparisons['multi']['tokens']}个token均整数一致；context和解码动作按原atol2e-5/rtol2e-4通过。独立训练后的任务成绩不要求相等。",'',
    '结果仍不等同于完整45任务MetaWorld论文基准，没有新增训练随机种子的重复实验。所有候选、失败示范、损失、逐步梯度、逐场景动作和较差结果都保留。实验任务完成不意味着一定达到论文成功率。']
args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(14,4))
for framework,matrix,color,offset in [('Torch',records,'#2d8793',-.18),('Jittor',native_matrix,'#dc8d35',.18)]:
    for i,(name,r) in enumerate(matrix.items()):
        lo,hi=r['wilson95'];x=i+offset
        axes[0].bar(x,r['rate'],width=.35,color=color,label=framework if i==0 else None)
        axes[0].errorbar(x,r['rate'],yerr=[[r['rate']-lo],[hi-r['rate']]],fmt='none',color='black',capsize=2)
        axes[0].text(x,r['rate']+.025,str(r['successes'])+'/50',ha='center',fontsize=6)
axes[0].legend(fontsize=7)
axes[0].set_xticks(range(4),['80/base','280/base','80/tuned','280/tuned'],rotation=20);axes[0].set_title('Fixed-update2x2:50new scenes')
for i,name in enumerate(['torch','jittor']):
    vals=[multi[name]['tasks'][task]['rate'] for task in protocol['multi_task_tasks']]
    axes[1].bar(np.arange(3)+(i-.5)*.35,vals,width=.35,label=name)
axes[1].set_xticks(range(3),['Reach','Reach-wall','Push']);axes[1].set_title('Shared3task policy:20scenes/task');axes[1].legend()
for name in records:
    rows=[json.loads(line) for line in (out/name/'metrics.jsonl').read_text().splitlines()]
    axes[2].plot([r['step'] for r in rows],[r['validation_ce'] for r in rows],label=name)
axes[2].set_title('Torch held-out validationCE');axes[2].set_xlabel('Updates');axes[2].legend(fontsize=7)
for ax in axes[:2]:ax.set_ylim(0,1.05);ax.set_ylabel('Task success rate');ax.grid(axis='y',alpha=.2)
fig.suptitle('Overnight data/tuning/multitask experiments;one training seed/config')
fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,(name,r) in zip(axes.flat,records.items()):
    native_sub=Path('selected_native') if name==native else Path('native_factorial')/name
    for framework,sub,color in [('Torch',Path(name),'#2d8793'),('Jittor',native_sub,'#dc8d35')]:
        rows=[json.loads(line) for line in (out/sub/'metrics.jsonl').read_text().splitlines()]
        x=[row['step'] for row in rows]
        ax.plot(x,[row['train_ce'] for row in rows],color=color,linestyle='--',label=framework+' train')
        ax.plot(x,[row['validation_ce'] for row in rows],color=color,label=framework+' validation')
    ax.set_title(name);ax.set_xlabel('Updates');ax.set_ylabel('Cross entropy');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Actual controlled training logs;best checkpoint selected by validation')
fig.tight_layout();fig.savefig(args.output.with_name(args.output.stem+'_loss_curves.png'),dpi=160);plt.close(fig)
print(json.dumps({'factorial':{k:r['successes'] for k,r in records.items()},'selected':native,'native_successes':native_record['successes'],
    'multi':{name:{task:r['successes'] for task,r in v['tasks'].items()} for name,v in multi.items()},'audit':audit},indent=2))
