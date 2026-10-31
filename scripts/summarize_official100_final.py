"""Report frozen final outcomes, all development variants and limitations."""
import argparse
import csv
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
job=args.bundle/'outputs/official100_policy_20261028'
workflow=json.loads((job/'final_workflow.json').read_text())
assert workflow['status']=='completed'
protocol=workflow['protocol'];records={}
data=args.bundle/'outputs/official_reach_100_20261028/adapted'
def wilson(k,n):
    p=k/n;z=1.959963984540054;d=1+z*z/n
    center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-radius,center+radius]
for name in ('torch','jittor'):
    result=workflow['results'][name]
    assert result['seed_set']=='final' and [r['seed'] for r in result['episodes']]==list(range(1000,1020))
    assert result['horizon']==protocol['choice']['horizon'] and result['top_k']==protocol['choice']['top_k']
    folder=data/'outputs'/f'official100_policy_20261028_{name}'
    summary=json.loads((folder/'summary.json').read_text())
    with (folder/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    best=min(rows,key=lambda r:float(r['validation_ce']))
    k=sum(bool(r['success']) for r in result['episodes'])
    records[name]={'successes':k,'episodes':20,'success_rate':k/20,'wilson95':wilson(k,20),
        'best_epoch':int(best['epoch']),'best_validation_ce':float(best['validation_ce']),
        'epochs_trained':len(rows),'training_seconds':summary['elapsed_seconds'],
        'median_decision_ms_unstandardized':result['median_decision_ms'],'episode_records':result['episodes']}
paired_agreement=sum(a['success']==b['success'] for a,b in zip(records['torch']['episode_records'],records['jittor']['episode_records']))
report={'scope':'small-scale single-task reproduction,not full paper benchmark','protocol':protocol,'frameworks':records,
    'paired_outcome_agreements':paired_agreement,'same_weights_greedy':json.loads((job/'same_weights_comparison.json').read_text()),
    'same_weights_top5':json.loads((job/'sampling5_comparison.json').read_text())}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
lines=['# 单任务小规模复现：最终测试结果（2026-10-28）','',
    '**原生Jittor和PyTorch均在首次保留测试中完成11/20个场景，成功率55%。训练、实现对齐和真实闭环仿真已跑通，效果仍有提升空间；不是原论文完整基准复现。**','',
    '## 最终结果','',
    '| 框架 | 成功/总数 | 成功率 | 95% Wilson区间 | 最佳验证CE（轮数） | 策略训练耗时 |',
    '|---|---:|---:|---:|---:|---:|']
for name,r in records.items():
    lines.append(f"| {name} | {r['successes']}/20 | {r['success_rate']:.0%} | {r['wilson95'][0]:.1%}–{r['wilson95'][1]:.1%} | {r['best_validation_ce']:.4f}（{r['best_epoch']}） | {r['training_seconds']:.1f}秒 |")
lines+=['',f"两个框架在{paired_agreement}/20个场景上的成功/失败标志相同。成功率相同不代表每个场景或动作轨迹相同，也不足以证明两个框架的性能等价。样本较少，不能据这次结果判定框架优劣。",'',
    '## 测试前固定的方案','',
    '使用100条作者脚本采集的Reach示范，共5389步；50组初始状态，每组两条带噪轨迹。训练40组80条，验证10组20条，无相同起点跨集合。两边从同一策略初始化训练20轮，按验证CE选最佳检查点。两边最佳均在较早轮次，后续验证误差上升，存在过拟合迹象。','',
    '固定最终场景1000–1019、最多150步、每次预测32步动作并执行8步、top-5随机采样、温度1、NumPy PCG64采样种子30000。所有选择仅使用验证集及开发场景2000–2009；final_protocol.json在最终测试前写入，记录模型/代码哈希，测试后核对未变，未根据最终结果继续调参。','',
    '## 开发集探索记录（不属于最终测试）','',
    '| 部署方式 | PyTorch | Jittor |','|---|---:|---:|']
for c in protocol['candidates']:
    lines.append(f"| {c['label']} | {c['development_successes']['torch']}/10 | {c['development_successes']['jittor']}/10 |")
lines+=['','top-5最终使用相同NumPy采样方式，避免把两个随机数实现的差异混入框架比较。最初Torch自带multinomial采样的额外探索结果2/10也完整保留，但不混入统一采样方式的选择表。上表参与开发调试，不能当作独立泛化证据。','',
    '## 分词器与独立思考','',
    '新分词器微调降低了带噪重建误差，但对无噪专家动作参考的误差从旧模型0.04398升到0.06105。因此本轮保留旧分词器，使用新数据训练条件策略。说明更低的原始Loss未必意味着更适合控制；随机扰动与可复用动作结构需要区分。这个额外离线诊断不是原论文默认选择规则，参考状态不进入策略。','',
    '100条轨迹只有50组初始状态，单纯重复同样起点不能等同于扩大任务覆盖。只把训练轮数加大也未必有效，因为验证最优出现在早期。下一轮更值得控制变量比较不同起点覆盖、正则化以及同协议24vs80训练子集；这批最终测试场景已使用，后续不得作为新的独立测试或调参依据。','',
    '## 实现对齐与证据','',
    '新数据43112个分词器标签在Torch/Jittor中全部相同。同一组训练后原生策略权重，在40个验证输入上的greedy320个token一致，context最大差约1.16e-6、动作最大差约4.92e-7；top-5使用相同随机数时的320个token也一致。原生Jittor训练及评估进程未导入Torch。独立优化得到不同参数和轨迹是另一个问题，不能把实现前向对齐等同于全部训练权重严格相同。','',
    '原始日志、逐场景动作/token、最佳模型、曲线与冻结协议位于 reports/official100_policy_completed_20261028/。完整模型归档302331729字节，SHA256 266d7eff7643eea3d7fc3c8aeec4a5d030fcd6e0bed7f43f43117be824a1295d。最终测试及消融补充归档SHA256 b92390929da3c02a85af9525654f56f0071c93c0322ee980233250a9e18872cd；两者均完成传输哈希和ZIP完整性校验。','',
    '绘图环节曾因Torch环境缺少matplotlib停止，改用已有绘图库的独立环境后从失败环节恢复，没有重新训练或删除失败记录。静态日志模板里的旧24/6计数保持原样以留存证据，并用protocol_snapshot中的实际80/20划分和批次计划校正说明。','',
    '与论文仍有显著区别：只做Reach单任务，没有45任务预训练和5任务少样本迁移；训练轮数及数据规模缩小，最终时限150步；没有把论文完整成功率当作对标成绩。','',
    '## 最终每个场景','',
    '| Seed | Torch成功/步数 | Jittor成功/步数 |','|---|---:|---:|']
for a,b in zip(records['torch']['episode_records'],records['jittor']['episode_records']):
    lines.append(f"| {a['seed']} | {a['success']} / {a['steps']} | {b['success']} / {b['steps']} |")
args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,name in enumerate(records):
    axes[0].bar([j+(i-.5)*.35 for j in range(3)],[c['development_successes'][name]/10 for c in protocol['candidates']],width=.35,label=name)
    r=records[name];p=r['success_rate'];lo,hi=r['wilson95']
    axes[1].bar(i,p,color=['#2d8793','#dc8d35'][i]);axes[1].errorbar(i,p,yerr=[[p-lo],[hi-p]],fmt='none',color='black',capsize=5)
    axes[1].text(i,p+.025,f"{r['successes']}/20",ha='center')
axes[0].set_xticks(range(3),['Greedy / execute8','Greedy / execute4','Top5 / execute8']);axes[0].tick_params(axis='x',rotation=12);axes[0].legend()
axes[0].set_title('Development exploration (10 scenes)');axes[1].set_title('Frozen final test (20 scenes; 95% CI)')
axes[1].set_xticks(range(2),['PyTorch','Native Jittor'])
for ax in axes:ax.set_ylim(0,1);ax.set_ylabel('Task success rate');ax.grid(axis='y',alpha=.2)
fig.suptitle('Single-task Reach reproduction — not full paper benchmark')
fig.tight_layout();fig.savefig(args.output.parent/'figures/official100_final_results.png',dpi=170);plt.close(fig)
print(json.dumps({name:{k:v for k,v in r.items() if k!='episode_records'} for name,r in records.items()},indent=2))
