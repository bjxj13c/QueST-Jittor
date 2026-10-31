"""Plot measured frozen-tokenizer transfer; reconstruction is not rollout."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
old=json.loads((args.input/'old30_jittor/metrics.json').read_text())
new=json.loads((args.input/'new100_jittor/metrics.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4))
labels=['Old30 valid','New100 valid','New100 mean baseline']
values=[old['splits']['validation']['mae_valid'],new['splits']['validation']['mae_valid'],new['splits']['validation']['train_mean_baseline_valid']]
bars=axes[0].bar(labels,values,color=['#27867d','#e49343','#9aa4b2'])
axes[0].bar_label(bars,fmt='%.4f',padding=3)
axes[0].set_ylim(0,.25);axes[0].set_ylabel('MAE on valid action positions')
axes[0].tick_params(axis='x',labelrotation=15)
axes[0].set_title('Same frozen tokenizer; collection protocol changed')
axes[1].bar(np.arange(4),new['splits']['validation']['mae_by_dimension'],color='#e49343')
axes[1].set_xticks(np.arange(4),['x','y','z','gripper'])
axes[1].set_ylabel('New100 validation MAE');axes[1].set_ylim(0,.22)
axes[1].set_title('Error by action dimension')
fig.suptitle('Offline action reconstruction — not robot task success')
fig.tight_layout();fig.savefig(args.output/'tokenizer_transfer_metrics.png',dpi=160);plt.close(fig)
with np.load(args.input/'new100_jittor/validation.npz') as data:
    first=[i for i,start in enumerate(data['start']) if start==0][:3]
    fig,axes=plt.subplots(len(first),4,figsize=(13,7),sharex=True,sharey=True,squeeze=False)
    for row,index in enumerate(first):
        valid=int(data['mask'][index].sum())
        for dim in range(4):
            ax=axes[row,dim]
            ax.plot(np.arange(valid),data['target'][index,:valid,dim],color='#315aa4',label='Noisy expert action',lw=1.5)
            ax.plot(np.arange(valid),data['prediction'][index,:valid,dim],color='#d47a25',label='Frozen reconstruction',lw=1.5)
            ax.set_title(f"{data['episode'][index]} / {['x','y','z','gripper'][dim]}")
            if row==len(first)-1:ax.set_xlabel('Future action step')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First three held-out episodes, start=0; future actions are model inputs')
    fig.tight_layout();fig.savefig(args.output/'tokenizer_transfer_reconstructions.png',dpi=150);plt.close(fig)
print('Saved frozen-tokenizer metrics and reconstruction plots.')
