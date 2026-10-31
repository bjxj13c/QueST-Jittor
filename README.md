# QueST-Jittor

> QueST 两阶段机器人模仿学习方法的原生 Jittor 复现

本项目基于 NeurIPS 2024 论文 [QueST: Self-Supervised Skill Abstractions for Learning Continuous Control](https://arxiv.org/abs/2407.15840)，实现动作分词器与条件策略，并保留 PyTorch 版本进行对照。

本次在 MetaWorld 的五个任务上，使用450条训练示范和100条验证示范，完成数据准备、两阶段训练、计算对齐及闭环测试。论文的完整多任务与少样本适配实验留待进一步开展。

- [原论文代码](https://github.com/pairlab/QueST)
- [论文项目页](https://quest-model.github.io/)
- [Jittor文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
- [迁移与对齐细节](details.md)
- [English README](README_en.md)

## 一、环境搭建

主要训练与评估环境如下。

| 项目 | 配置 |
|---|---|
| 操作系统 / Python | Ubuntu 22.04 / Python 3.10.22 |
| GPU / CUDA | RTX 4090D 24GB / CUDA 12.1 |
| Jittor | 1.3.11.0 |
| PyTorch / torchvision | 2.2.0＋cu121 / 0.17.0＋cu121 |
| MuJoCo / Gymnasium | 2.3.7 / 0.29.1 |
| MetaWorld | 固定版本，见 [依赖文件](requirements/simulation.txt) |

以下命令在仓库根目录执行，建议分别建立 Jittor 和 PyTorch 环境。系统需已有 Python 3.10、g++ 和 CUDA 工具链。

```bash
bash scripts/bootstrap_linux.sh jittor /path/to/python3.10 cpu
bash scripts/bootstrap_linux.sh torch /path/to/python3.10 cu121

.venv-jittor-linux/bin/python -m pip install -r requirements/simulation.txt
.venv-torch-linux/bin/python -m pip install -r requirements/simulation.txt -c requirements/cpu.txt
```

无桌面服务器使用 EGL 渲染：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_TF32_OVERRIDE=0

.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cuda \
  scripts/jittor_cpu_probe.py --device cuda --output reports/jittor_cuda_probe.json
```

Jittor由运行入口选择CPU或CUDA，安装参数中的`cpu`不限制后续GPU计算。安装检查和常见环境问题见[环境说明](docs/INSTALL.md)。已有Python 3.10环境也可通过`pip install -r requirements.txt`安装Jittor与仿真依赖。

## 二、数据与模型

五个任务分别为 Reach、Reach-wall、Push、Door-open、Window-open。示范由 MetaWorld 脚本专家采集，每任务90条训练、20条验证；按完整轨迹与目标位置划分。

| 数据 | 单条轨迹的格式 |
|---|---|
| RGB图像 | `[T, 128, 128, 3]`，uint8 |
| 机器人状态 | `[T, 8]` |
| 专家动作 | `[T, 4]` |
| 奖励与成功标志 | 每次执行动作后的环境反馈 |

策略使用RGB、8维机器人状态和任务ID。连续32步动作经分词器编码为8个token，再由解码器重建；执行时每轮采用前8步动作，然后重新观察。

本次实验的数据与权重尚未提供在线下载，复查已报告结果需先恢复对应材料。文件结构、初始化来源及重新采集方法见[数据准备](docs/DATA.md)和[运行补充说明](docs/REPRODUCTION_GUIDE.md)。

```text
outputs/five_task_10pct_20261029/
├── dataset/
│   ├── data/demonstrations.hdf5
│   ├── records/reach_split_v1.json
│   └── assets/                   # 冻结分词器、配置与token标签
└── experiments/
    ├── initial_five.npz           # 公共策略初始化
    ├── policy_torch/
    └── policy_jittor/
```

## 三、训练与测试

### 1. 训练

先训练动作分词器，固定分词器并生成token标签，再训练条件策略。根目录提供以下入口，参数继续使用原训练脚本的格式。

| 入口 | 用途 |
|---|---|
| `train_jittor.py --stage tokenizer` | 原生Jittor动作分词器 |
| `train_jittor.py --stage policy` | 原生Jittor条件策略 |
| `train_pytorch.py --stage tokenizer` | PyTorch分词器参考 |
| `train_pytorch.py --stage policy` | PyTorch条件策略参考 |

以已恢复分词器、标签和公共初始化后的Jittor策略训练为例：

```bash
QUEST_DATA=outputs/five_task_10pct_20261029/dataset
QUEST_RUN=outputs/five_task_10pct_20261029/experiments

.venv-jittor-linux/bin/python train_jittor.py --stage policy --device cuda \
  --root "$QUEST_DATA" --output "$QUEST_DATA/outputs/readme_policy_jittor" \
  --tokenizer-weights "$QUEST_DATA/assets/weights.npz" \
  --tokenizer-config "$QUEST_DATA/assets/config.json" \
  --canonical-labels "$QUEST_DATA/assets/canonical_labels" \
  --initial-weights "$QUEST_RUN/initial_five.npz" \
  --updates 12000 --eval-every 1000 --lr 0.0001 --weight-decay 0.0001 \
  --dropout 0 --shift 0 --seed 310500
```

每批16个样本，条件策略完成12,000次参数更新，按验证集交叉熵保存模型。五任务主基线的分词器从已有权重继续训练10轮；后续激活函数实验使用全新初始化。分词器训练、标签生成和PyTorch对照的完整步骤见[训练说明](docs/TRAINING.md)。

### 2. 测试

下面复核已保存五任务模型的开门任务。输入模型、分词器与配置应来自同一组实验。

```bash
QUEST_DATA=outputs/five_task_10pct_20261029/dataset
QUEST_RUN=outputs/five_task_10pct_20261029/experiments

.venv-jittor-linux/bin/python test_jittor.py --device cuda \
  --root "$QUEST_DATA" \
  --policy-weights "$QUEST_RUN/policy_jittor/best_weights.npz" \
  --policy-config "$QUEST_RUN/policy_jittor/config.json" \
  --tokenizer-weights "$QUEST_DATA/assets/weights.npz" \
  --tokenizer-config "$QUEST_DATA/assets/config.json" \
  --env-name door-open-v2-goal-observable --task-id 3 \
  --seed-set experiment --seed-start 12000 --episodes 30 \
  --per-episode-sampling --top-k 5 --sampling-seed 30000 \
  --max-steps 150 --action-horizon 8 --videos 0 --name readme_door_jittor
```

每个任务测试30个场景，最多运行150步，以环境返回的成功标志计算成功率。其他任务、PyTorch测试及输出格式见[测试说明](docs/EVALUATION.md)。`--dry-run`可查看根目录入口将调用的命令，不开始训练或测试。

## 四、实验结果

### 1. 五任务基线

两框架使用相同的数据、初始化和训练预算，策略共享冻结分词器及token标签。以下是一组训练种子的结果。

| 任务 | PyTorch成功次数 /30 | Jittor成功次数 /30 |
|---|---:|---:|
| Reach | 13 | 12 |
| Reach-wall | 27 | 24 |
| Push | 10 | 9 |
| Door-open | 29 | 28 |
| Window-open | 29 | 29 |
| **合计** | **108/150（72%）** | **102/150（68%）** |

![五任务基线结果](reports/FIVE_TASK_10PCT_RESULTS_20261029.png)

### 2. 计算对齐

使用同一份训练权重和200组验证输入，逐项比较模型输出。

| 检查项 | 结果 |
|---|---:|
| 条件特征最大绝对误差 | `1.1772e-6` |
| 1,600个动作token | 全部一致 |
| 解码动作最大绝对误差 | `1.8418e-5` |
| 浮点误差标准 | `abs(error) <= 2e-5 + 2e-4 * abs(reference)`，通过 |

前向、梯度和短程更新的实现与记录见[迁移细节](details.md)，五任务完整结果见[实验报告](reports/FIVE_TASK_10PCT_RESULTS_20261029.md)。

### 3. 改进尝试

| 实验 | 主要结果 |
|---|---|
| 12组激活函数筛选 | 本轮Mish＋ReLU平均67.67%，Mish＋GELU为62%；同轮共享全新初始化 |
| 三种子独立验证 | 两框架平均均提高2个百分点，95%区间包含0，尚不足以确认稳定收益 |
| 保持示范 | 部分配置的Reach保持率提高，五任务总体成功率未一致提高 |
| 反馈间隔、均衡采样、视觉预训练 | 在各自对照设置下未观察到一致的整体收益 |

各轮的初始化、场景与统计范围分别说明，详见[扩展实验](docs/EXPERIMENTS.md)。原始失败及保持补测的独立数值修复记录保留在[收尾报告](reports/EXPERIMENT_CLOSEOUT_20261030.md)。

### 4. 推理耗时

同一GPU、FP32，预热10次、测量50次并同步；不包含环境渲染。

| 项目 | PyTorch中位耗时 | Jittor中位耗时 |
|---|---:|---:|
| 分词器编码、量化与解码，batch=16 | 7.234 ms | 8.295 ms |
| 策略生成与动作解码，batch=1、greedy | 25.162 ms | 32.815 ms |

该结果对应激活对照中的Mish＋GELU检查点。在线评估使用top-5采样；并发训练耗时与独占推理计时分别记录。[测速设置](docs/EXPERIMENTS.md#日志曲线和性能)

## 五、执行演示

左列为Jittor学习策略，右列为MetaWorld脚本专家；每对使用相同初始场景。下表链接均为成功执行片段。

| 任务 | Jittor学习策略 | 脚本专家 |
|---|---|---|
| Reach | [64步成功](reports/demos/five_task_pairs_20261031/reach-v2__learned_jittor__seed12000__720p.mp4) | [47步成功](reports/demos/five_task_pairs_20261031/reach-v2__scripted_expert__seed12000__720p.mp4) |
| Reach-wall | [48步成功](reports/demos/five_task_pairs_20261031/reach-wall-v2__learned_jittor__seed12000__720p.mp4) | [50步成功](reports/demos/five_task_pairs_20261031/reach-wall-v2__scripted_expert__seed12000__720p.mp4) |
| Push | [71步成功](reports/demos/five_task_pairs_20261031/push-v2__learned_jittor__seed12001__720p.mp4) | [66步成功](reports/demos/five_task_pairs_20261031/push-v2__scripted_expert__seed12001__720p.mp4) |
| Door-open | [76步成功](reports/demos/five_task_pairs_20261031/door-open-v2__learned_jittor__seed12000__720p.mp4) | [118步成功](reports/demos/five_task_pairs_20261031/door-open-v2__scripted_expert__seed12000__720p.mp4) |
| Window-open | [83步成功](reports/demos/five_task_pairs_20261031/window-open-v2__learned_jittor__seed12000__720p.mp4) | [87步成功](reports/demos/five_task_pairs_20261031/window-open-v2__scripted_expert__seed12000__720p.mp4) |

学习策略片段重放保存的动作，片尾1秒为标注定格。视频与原评估的核对说明见[演示目录](reports/demos/five_task_pairs_20261031/README.md)。

## 六、目录说明

```text
QueST-Jittor/
├── train_jittor.py        # Jittor两阶段训练入口
├── train_pytorch.py       # PyTorch参考训练入口
├── test_jittor.py         # Jittor闭环测试入口
├── requirements.txt      # Jittor与MetaWorld依赖
├── src/quest_repro/       # 模型、数据和训练工具
├── scripts/              # 采集、对齐、各轮实验及结果核验
├── requirements/         # 分环境依赖版本
├── records/              # 实验配置与协议
├── reports/              # 结果、曲线、轻量日志和视频
├── docs/                 # 安装、数据、训练、测试与扩展实验
├── tests/                # 已有单元检查
├── licenses/             # 第三方许可证
└── details.md            # 框架迁移与数值对齐细节
```

文档导航见[docs/README.md](docs/README.md)，完整历史实验见[实验索引](docs/EXPERIMENT_INDEX.md)。

## 七、参考与许可

- [QueST论文](https://arxiv.org/abs/2407.15840)与[官方代码](https://github.com/pairlab/QueST)
- [Jittor官方文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [DMANet-Jittor](https://github.com/Yuyciciccc/DMANet-Jittor)，参考其环境、数据、训练测试和结果分开说明的组织方式

本项目采用[MIT License](LICENSE)。改编实现的来源、固定版本及第三方许可证见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
