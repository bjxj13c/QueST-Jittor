"""Select top3+baseline,link paired five-task data,freeze warm-start checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--unit');a=p.parse_args();root=a.root.resolve();out=root/'outputs/selected_activation_holding_20261030';grid=root/'outputs/activation_grid_20261029'
    if a.unit:
        import torch
        from quest_repro.config import TokenizerConfig
        from quest_repro.tokenizer_torch import SkillVAE
        from quest_repro.training import load_npz_state
        from quest_repro.policy_data import PolicyDataset
        combo,arm=a.unit.split('__');assert combo==os.environ['QUEST_ACTIVATION_CONV']+'_'+os.environ['QUEST_ACTIVATION_FFN'];unit=out/'units'/a.unit;assets=unit/'assets';assets.mkdir(exist_ok=False)
        source=unit/'tokenizer_jittor';cfg=read(source/'config.json')['config'];best=read(source/'best_validation.json');torch.set_num_threads(2)
        tokenizer=SkillVAE(TokenizerConfig(**cfg)).cuda().eval();load_npz_state(tokenizer,source/'best_weights.npz')
        np.savez_compressed(assets/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in tokenizer.state_dict().items()})
        torch.save({'model':{k:v.detach().cpu() for k,v in tokenizer.state_dict().items()},'config':cfg,'step':best['step'],'extra':{'activation_case':combo,'holding_arm':arm}},assets/'tokenizer_shared.pt')
        (assets/'config.json').write_text(json.dumps({'config':cfg,'activation_case':combo,'holding_arm':arm,'selected_step':best['step']},indent=2))
        labels=assets/'canonical_labels';labels.mkdir()
        for split in ('train','validation'):
            data=PolicyDataset(unit/'dataset',split,tokenizer,assets/'weights.npz',batch_size=16,device='cuda');np.savez_compressed(labels/(split+'.npz'),labels=data.labels)
        return
    source=root/'reports/ACTIVATION_GRID_RESULTS_20261029.json';results=read(source);assert results['status']=='audited_complete_grid';rows=results['rows'];assert len(rows)==24 and all(r['status']=='completed' for r in rows)
    ranks=[]
    for name in results['protocol']['cases']:
        pair=[r for r in rows if r['case']==name];assert len(pair)==2
        assert read(grid/'cases'/name/'audit.json')['status']=='passed'
        ranks.append({'combo':name,'mean_success':float(np.mean([r['success_rate'] for r in pair])),'min_framework_success':float(min(r['success_rate'] for r in pair)),'mean_tokenizer_valid_mae':float(np.mean([r['tokenizer_mae_valid'] for r in pair]))})
    ranks.sort(key=lambda r:(-r['mean_success'],-r['min_framework_success'],r['mean_tokenizer_valid_mae'],r['combo']))
    selected=[r['combo'] for r in ranks[:3]]
    if 'mish_gelu' not in selected:selected.append('mish_gelu')
    shared=out/'shared';shared.mkdir(exist_ok=False);original=root/'outputs/five_task_10pct_20261029/dataset';m=read(original/'records/reach_split_v1.json');paired=root/'outputs/padding_hold_20261029/collection'
    files=[original/m['dataset'],paired/'repeat/data/reach.hdf5',paired/'holding/data/reach.hdf5'];external={q.relative_to(root).as_posix():sha(q) for q in files}
    original_train=[e for e in m['train'] if m['episode_task_ids'][e]==0];original_val=[e for e in m['validation'] if m['episode_task_ids'][e]==0];assert len(original_train)==90 and len(original_val)==20
    mapping={e:f'new_reach_{i:03d}' for i,e in enumerate(original_train+original_val)};templates={}
    for arm in ('repeat','holding'):
        container=shared/(arm+'.hdf5')
        with h5py.File(container,'x') as file:
            for e in m['train']+m['validation']:
                if e in mapping:
                    i=int(mapping[e].split('_')[-1]);target=paired/arm/'data/reach.hdf5';name=mapping[e];group=f'/demo_{i}'
                else:target=original/m['dataset'];name=e;group='/'+e
                file[name]=h5py.ExternalLink(os.path.relpath(target,container.parent),group)
        manifest=dict(m);manifest.update(sha256=sha(container),train=[mapping.get(e,e) for e in m['train']],validation=[mapping.get(e,e) for e in m['validation']],
            episode_task_ids={mapping.get(e,e):m['episode_task_ids'][e] for e in m['train']+m['validation']},external_source_sha256=external,
            scope='same4nonReach tasks;Reach replaced with paired90train20validation newprefixes or true60stepholding continuation')
        manifest['provenance']={'unchanged_four_tasks':m.get('provenance'),'reach_source':str(paired.relative_to(root)),'paired_goal_mapping':mapping,'split':'four task episode memberships unchanged;Reach90train20val fixed paired collection'}
        templates[arm]=manifest
        with h5py.File(container,'r') as f:
            assert len(manifest['train'])==450 and len(manifest['validation'])==100
            for e in manifest['train']+manifest['validation']:
                assert f[e]['images'].shape[0]==f[e]['actions'].shape[0]==f[e]['robot_states'].shape[0]
    init={}
    for combo in selected:
        base=grid/'cases'/combo;init[combo]={'tokenizer':str((base/'tokenizer_jittor/best_weights.npz').relative_to(root)),
            'tokenizer_config':str((base/'tokenizer_jittor/config.json').relative_to(root)),'policy':str((base/'policy_jittor/best_weights.npz').relative_to(root))}
        init[combo]['hashes']={k:sha(root/init[combo][k]) for k in ('tokenizer','tokenizer_config','policy')}
        for arm in ('repeat','holding'):
            unit=out/'units'/(combo+'__'+arm);data=unit/'dataset';(data/'records').mkdir(parents=True)
            manifest=dict(templates[arm]);manifest['dataset']=os.path.relpath(shared/(arm+'.hdf5'),data).replace('\\','/')
            (data/'records/reach_split_v1.json').write_text(json.dumps(manifest,indent=2))
    record={'status':'selected_before_new_training','grid_report_sha256':sha(source),'ranking':ranks,'selected':selected,'selection_rule':'top3bymeanTorch/Jittor5tasksuccess;tiesminframeworksuccess,meanvalidMAE,name;alwaysaddbaseline',
        'initializations':init,'external_sources':external,'data_containers':{arm:sha(shared/(arm+'.hdf5')) for arm in templates},'arms':['repeat','holding'],'units':[c+'__'+arm for c in selected for arm in ('repeat','holding')]}
    (out/'selection.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
