"""Create a report strictly from completed official100 workflow evidence."""
import argparse
import csv
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--figure',type=Path,required=True)
args=parser.parse_args()
job=args.bundle/'outputs/official100_policy_20261028'
workflow=json.loads((job/'workflow.json').read_text())
assert workflow['status']=='completed'
data=args.bundle/'outputs/official_reach_100_20261028/adapted'
records={}
def interval(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-r,c+r]
for name in ('torch','jittor'):
    folder=data/'outputs'/f'official100_policy_20261028_{name}'
    summary=json.loads((folder/'summary.json').read_text())
    dev=json.loads((folder.with_name(folder.name+'_dev10')/'summary.json').read_text())
    with (folder/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    assert summary['status']=='complete' and len(rows)==20 and dev['seed_set']=='development'
    best=min(rows,key=lambda r:float(r['validation_ce']))
    k=sum(bool(ep['success']) for ep in dev['episodes']);n=len(dev['episodes'])
    records[name]={'epochs':len(rows),'steps':int(rows[-1]['step']),
        'best_epoch':int(best['epoch']),'best_ce':float(best['validation_ce']),
        'training_seconds':summary['elapsed_seconds'],'successes':k,'episodes':n,
        'interval':interval(k,n),'episode_records':dev['episodes'],
        'first5_successes':sum(bool(ep['success']) for ep in dev['episodes'] if ep['seed']<2005),
        'rows':rows}
result={'scope':workflow['scope'],'tokenizer_selection':workflow['tokenizer_selection'],
        'same_weights':workflow['same_weights_comparison'],'frameworks':records}
args.report.parent.mkdir(parents=True,exist_ok=True)
args.report.with_suffix('.json').write_text(json.dumps(result,indent=2))
lines=['# 新100条数据条件策略实验结果','',
    '本轮为单任务Reach的小规模复现：80条训练、20条验证，按初始状态分组，两边无重复起点。两个框架使用相同初始化、动作分词器、批次顺序，分别训练20轮。最终保留测试种子1000–1019未使用。','',
    '| 框架 | 最佳验证CE / 轮数 | 训练耗时 | 开发场景成功数 | 成功率95% Wilson区间 |',
    '|---|---:|---:|---:|---:|']
for name,r in records.items():
    lines.append(f"| {name} | {r['best_ce']:.4f} / {r['best_epoch']} | {r['training_seconds']:.1f}秒 | {r['successes']}/{r['episodes']} | {r['interval'][0]:.1%}–{r['interval'][1]:.1%} |")
lines+=['','## 分词器选择与实验解释','',
    f"旧分词器对无噪专家参考的验证MAE为{workflow['tokenizer_selection']['old_clean_reference_mae']:.5f}，新分词器为{workflow['tokenizer_selection']['new_clean_reference_mae']:.5f}。本轮按该离线诊断选择旧分词器并冻结，用新数据训练条件策略；不是宣称新分词器训练失败。这项选择是额外实验设计，不能当作原论文完整协议。39维完整状态仅用于专家诊断，策略输入始终只有图像和8维机器人状态。",'',
    '开发场景为2000–2009，greedy top1，每次生成32步动作并执行前8步，再获取反馈，最多150步。它与论文的完整多任务、训练时长及随机top5采样协议有差异。开发场景用于诊断，样本少且可能反复使用，不能替代最终评估或据此判定框架孰优。旧30条与新100条同时改变采集协议及数据量，不能只归因于样本数量。','',
    '## 独立优化与同权重实现对齐','',
    '同一组训练后的原生Jittor策略权重，在独立Torch/Jittor进程的验证输入上比较。原始结果如下；独立训练得到不同参数或不同场景结果，不等于同权重实现不一致。','',
    '```json',json.dumps(workflow['same_weights_comparison'],indent=2),'```','',
    '## 每个开发场景的结果','',
    '| Seed | Torch成功/步数 | Jittor成功/步数 |','|---|---:|---:|']
for a,b in zip(records['torch']['episode_records'],records['jittor']['episode_records']):
    assert a['seed']==b['seed']
    lines.append(f"| {a['seed']} | {a['success']} / {a['steps']} | {b['success']} / {b['steps']} |")
lines+=['','完整日志、最佳权重、逐步动作/token记录、训练曲线和示例视频随归档保存。所有数据来自实际运行，没有将专家成绩当作学习策略成绩。']
args.report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4))
for name,r in records.items():
    epochs=[int(row['epoch']) for row in r['rows']]
    axes[0].plot(epochs,[float(row['train_ce']) for row in r['rows']],label=name+' train')
    axes[0].plot(epochs,[float(row['validation_ce']) for row in r['rows']],linestyle='--',label=name+' validation')
    axes[1].plot(epochs,[float(row['autoregressive_valid_mae']) for row in r['rows']],label=name)
for i,(name,r) in enumerate(records.items()):
    p=r['successes']/r['episodes'];lo,hi=r['interval']
    axes[2].bar(i,p,color=['#2d8793','#dc8d35'][i]);axes[2].errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='none',color='black',capsize=4)
    axes[2].text(i,p+.04,f"{r['successes']}/{r['episodes']}",ha='center')
axes[0].set_title('Train / held-out cross entropy');axes[1].set_title('Autoregressive valid action MAE')
axes[2].set_title('Development success; 95% Wilson CI');axes[2].set_xticks([0,1],['Torch','Jittor']);axes[2].set_ylim(0,1.1)
for ax in axes[:2]:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Single-task Reach: new100 data, shared frozen tokenizer')
fig.tight_layout();args.figure.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.figure,dpi=160);plt.close(fig)
print(json.dumps({name:{k:v for k,v in r.items() if k!='rows'} for name,r in records.items()},indent=2))
