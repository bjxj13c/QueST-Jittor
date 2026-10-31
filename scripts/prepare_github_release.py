"""Prepare an isolated public source tree without changing the research Git index."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'exports/github_ready/QueST-Jittor'
EXCLUDE={'remote_session.py','clone_remote_session.py','resume_closeout_transfers.py','stream_backup_chunk.py','verify_resumable_closeout.py'}


def main():
    assert not DEST.exists(),'Use a new output directory; never erase an earlier release'
    DEST.mkdir(parents=True)
    selected=set()
    for folder in ['src','tests','requirements','licenses']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:selected.add(p)
    for p in (ROOT/'scripts').glob('*'):
        if p.is_file() and p.name not in EXCLUDE:selected.add(p)
    for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md','pyproject.toml','.gitattributes']:
        if (ROOT/name).exists():selected.add(ROOT/name)
    for name in ['REPRODUCTION_GUIDE.md','EXPERIMENT_INDEX.md','服务器选择与答辩说明.md']:
        selected.add(ROOT/'docs'/name)
    for p in (ROOT/'records').glob('*.json'):
        if 'protocol' in p.name or p.name in ['reach_split_v1.json','data_manifest_v1.json']:selected.add(p)
    for name in ['ACTIVATION_CONFIRMATION_20261030.json','ACTIVATION_CONFIRMATION_20261030.md','ACTIVATION_CONFIRMATION_REVIEW_20261031.md','ACTIVATION_CONFIRMATION_LOCAL_VERIFICATION_20261031.json','ACTIVATION_CONFIRMATION_ARCHIVE_VERIFICATION_20261031.json','ACTIVATION_CONFIRMATION_BACKUP_FINAL_VERIFICATION_20261031.json','ACTIVATION_CONFIRMATION_20261031.png','final_cpu_tests_20261031.xml']:
        selected.add(ROOT/'reports'/name)
    # Include the final paired video examples, not failed export attempts.
    videos=ROOT/'reports/demos/five_task_pairs_20261031'
    proof=json.loads((videos/'verification.json').read_text(encoding='utf-8'))
    for row in proof['videos']:selected.add(videos/row['video'])
    for name in ['README.md','verification.json','local_video_review.json','preview_sheet.jpg']:selected.add(videos/name)
    # Traverse actual documentation links, preserving linked lightweight reports.
    done=set()
    while True:
        pending=[p for p in selected if p.suffix=='.md' and p not in done]
        if not pending:break
        for p in pending:
            done.add(p)
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                if '://' in target or target.startswith('#'):continue
                q=(p.parent/target.split('#')[0]).resolve()
                if not q.is_relative_to(ROOT) or not q.exists():continue
                if q.is_file():selected.add(q)
                elif q.is_dir() and q.is_relative_to(ROOT/'reports/runs'):
                    selected.update(v for v in q.rglob('*') if v.is_file() and v.suffix in ['.json','.csv','.jsonl','.txt'])
    files=[]
    forbidden={'.pt','.pth','.pkl','.npz','.hdf5','.zip','.gz','.key','.pem'}
    patterns={'private_key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
              'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b'),
              'huggingface_token':re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
              'credential_url':re.compile(r'https?://[^\s/:@]+:[^\s/@]+@')}
    for p in sorted(selected):
        assert p.is_file(),str(p)
        relative=p.relative_to(ROOT)
        assert p.suffix not in forbidden and p.stat().st_size<15_000_000,str(relative)
        if p.suffix not in {'.png','.jpg','.jpeg','.mp4','.pdf'}:
            text=p.read_text(encoding='utf-8',errors='replace')
            for kind,pattern in patterns.items():assert not pattern.search(text),(str(relative),kind)
        q=DEST/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);files.append({'path':relative.as_posix(),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    (DEST/'.gitignore').write_text('''# Environments, data and local operational artifacts
.venv*/
cache/
upstream/
backups/
data/
data_runs/
outputs/
exports/
deliverables/
work_*/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
*.pt
*.pth
*.pkl
*.npz
*.hdf5
*.zip
*.tar.gz
.env
*.pem
*.key
records/remote_session/
reports/five_task_evidence_*/
reports/*BACKUP_QUEUE*
reports/*TRANSFER_STATUS*
''',encoding='utf-8')
    (DEST/'docs/PUBLISHING.md').write_text('''# 首次发布

本目录是独立整理的公开源码候选，尚未创建GitHub仓库，也没有推送。
源码、实验协议、轻量日志、结果与演示视频已整理；不含训练数据、模型权重、远程连接队列或本机备份。

## 本机检查

准备此版本的原工作区CPU测试为27项通过，见reports/final_cpu_tests_20261031.xml。
该测试依赖本机保留的30条数据和固定官方源码；这不是“空仓库克隆即可跑完全部测试”的证明。
只验证不依赖采集数据的工具，可以安装依赖后运行：

```bash
python -m pytest tests/test_numeric_comparator.py tests/test_sampling.py -q
```

完整模型参考测试还需在upstream/QueST放置固定官方提交cc25a835a00cf2d91df69704bdd757904d13c061，并准备对应数据；见运行说明。
当前尚未提供公开数据／权重下载地址，也尚未做干净Linux环境的端到端复跑，不应在发布说明中声称已经完成。

## 创建和推送

在GitHub创建空仓库QueST-Jittor，不额外生成README或许可证。
使用你已有的安全登录方式；Git用户名与邮箱使用本人真实设置，不把密码或token写到命令里。
确认暂存内容后执行首次commit，再把真实仓库地址设置为origin并push main。
本目录已经初始化本地Git并暂存，尚未commit；没有添加虚构邮箱或远程地址。
发布后在未登录窗口检查README和视频链接，再把真实地址加入PPT。
''',encoding='utf-8')
    (DEST/'CITATION.cff').write_text('''cff-version: 1.2.0
message: "If using this reproduction, cite the original QueST paper and describe this repository as a small-scale reproduction."
title: "QueST native Jittor small-scale reproduction"
type: software
authors:
  - family-names: "张"
    given-names: "鼎"
license: MIT
version: "0.1.0"
preferred-citation:
  type: article
  title: "QueST: Self-Supervised Skill Abstractions for Learning Continuous Control"
  authors:
    - family-names: Mete
      given-names: Atharva
    - family-names: Xue
      given-names: Haotian
    - family-names: Wilcox
      given-names: Albert
    - family-names: Chen
      given-names: Yongxin
    - family-names: Garg
      given-names: Animesh
  year: 2024
  url: "https://arxiv.org/abs/2407.15840"
''',encoding='utf-8')
    # Validate all public Markdown targets, except pure anchors/external URLs.
    missing=[]
    for p in DEST.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            if not (p.parent/target.split('#')[0]).exists():missing.append({'file':str(p.relative_to(DEST)),'target':target})
    report={'status':'prepared' if not missing else 'needs_link_review','payload_files':len(files),'payload_bytes':sum(v['bytes'] for v in files),'missing_links':missing,'credential_format_findings':0,'data_weights_included':False,'published':False,'original_worktree_index_unchanged':True}
    (ROOT/'reports/GITHUB_RELEASE_PREPARATION_20261031.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (DEST/'SOURCE_MANIFEST.json').write_text(json.dumps({'files':files,'scope':'copied source evidence; generated publishing metadata is additional'},indent=2),encoding='utf-8')
    assert not missing,missing
    subprocess.run(['git','init','-b','main',str(DEST)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(DEST),'add','.'],check=True,capture_output=True)
    archive=ROOT/'deliverables/20261031/QueST-Jittor_GitHub源码包_20261031.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in DEST.rglob('*'):
            if p.is_file() and '.git' not in p.relative_to(DEST).parts:z.write(p,Path('QueST-Jittor')/p.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({**report,'archive_bytes':archive.stat().st_size},indent=2))


if __name__=='__main__':main()
