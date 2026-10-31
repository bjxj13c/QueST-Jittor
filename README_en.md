# QueST-Jittor

> Native Jittor implementation of QueST for robot imitation learning

This project reproduces the core two-stage method from [QueST: Self-Supervised Skill Abstractions for Learning Continuous Control](https://arxiv.org/abs/2407.15840), NeurIPS 2024. It includes an action tokenizer, a conditional policy, a PyTorch reference, and controlled experiments on five MetaWorld tasks.

The reported baseline uses 450 training and 100 validation demonstrations. The full multi-task and few-shot adaptation benchmarks are future work.

- [中文说明](README.md)
- [Official implementation](https://github.com/pairlab/QueST)
- [Project page](https://quest-model.github.io/)
- [Jittor documentation](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
- [Implementation and alignment details](details.md)

## 1. Environment

| Component | Experiment environment |
|---|---|
| OS / Python | Ubuntu 22.04 / Python 3.10.22 |
| GPU / CUDA | RTX 4090D 24GB / CUDA 12.1 |
| Jittor | 1.3.11.0 |
| PyTorch / torchvision | 2.2.0+cu121 / 0.17.0+cu121 |
| MuJoCo / Gymnasium | 2.3.7 / 0.29.1 |

Create separate project environments from an existing Python 3.10 installation:

```bash
bash scripts/bootstrap_linux.sh jittor /path/to/python3.10 cpu
bash scripts/bootstrap_linux.sh torch /path/to/python3.10 cu121

.venv-jittor-linux/bin/python -m pip install -r requirements/simulation.txt
.venv-torch-linux/bin/python -m pip install -r requirements/simulation.txt -c requirements/cpu.txt
```

The Jittor launcher selects CPU or CUDA explicitly. An existing compiler and CUDA toolkit are required for GPU runs. See [installation instructions](docs/INSTALL.md) for checks and EGL rendering setup.

## 2. Data and models

The tasks are Reach, Reach-wall, Push, Door-open, and Window-open. Each has 90 training and 20 validation demonstrations collected with MetaWorld scripted experts. Episodes and target positions are separated between splits.

The policy receives uint8 RGB images, an 8-dimensional robot state, and a task ID. A tokenizer encodes 32 four-dimensional actions into 8 tokens using FSQ levels `[8,5,5,5]`. The decoder reconstructs a 32-step action block; evaluation executes the first 8 steps and observes again.

Dataset and checkpoint downloads are not currently provided. Rechecking the reported results requires restoring the matching experiment files. [Data preparation](docs/DATA.md) and the [runbook](docs/REPRODUCTION_GUIDE.md) describe their structure, initializations, and collection routes.

## 3. Training and evaluation

| Entry point | Purpose |
|---|---|
| `train_jittor.py --stage tokenizer` | Native Jittor tokenizer training |
| `train_jittor.py --stage policy` | Native Jittor policy training |
| `train_pytorch.py --stage tokenizer` | PyTorch tokenizer reference |
| `train_pytorch.py --stage policy` | PyTorch policy reference |
| `test_jittor.py` | Native Jittor closed-loop evaluation |

The front ends dispatch to the existing scripts. Use the corresponding environment's Python interpreter. `--dry-run` prints the command without importing a model or starting a run.

After restoring the baseline data, frozen tokenizer, token labels, and shared initialization, policy training can be started with:

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

Train the tokenizer first, freeze it, and generate labels before policy training. Full instructions are in [TRAINING.md](docs/TRAINING.md), including PyTorch commands. The main baseline adapts an existing tokenizer for 10 epochs; the activation studies train from fresh shared initializations.

[EVALUATION.md](docs/EVALUATION.md) lists evaluation commands and task IDs. Each model is tested on 30 scenes per task, with a 150-step limit and the original environment success criterion.

## 4. Results

### Five-task baseline

Both frameworks use the same data, initial weights, training budget, frozen tokenizer, and token labels. Results below are from one training seed.

| Task | PyTorch successes /30 | Jittor successes /30 |
|---|---:|---:|
| Reach | 13 | 12 |
| Reach-wall | 27 | 24 |
| Push | 10 | 9 |
| Door-open | 29 | 28 |
| Window-open | 29 | 29 |
| **Total** | **108/150 (72%)** | **102/150 (68%)** |

### Same-weight alignment

On 200 validation inputs, with identical weights and sampling:

| Check | Result |
|---|---:|
| Maximum absolute context error | `1.1772e-6` |
| 1,600 action tokens | All identical |
| Maximum absolute decoded-action error | `1.8418e-5` |
| Floating-point criterion | `abs(error) <= 2e-5 + 2e-4 * abs(reference)`, passed |

See [details.md](details.md) for forward, gradient, update checks, and a separately recorded decoder precision repair in one holding-study branch.

### Additional experiments

- A 12-combination activation screen ranked Mish+ReLU first at 67.67% averaged across frameworks, versus 62% for Mish+GELU under that screen's fresh initialization and scenes.
- A separate confirmation with three new training seeds found a mean gain of 2 percentage points in each framework. Both 95% intervals include zero, so stable improvement is not established.
- Post-success expert demonstrations improved holding in some configurations, without consistent five-task success gains.
- Feedback interval, task-balanced sampling, and visual pretraining were assessed in their own paired experiments.

Full results, settings, logs, and limitations are linked from [EXPERIMENTS.md](docs/EXPERIMENTS.md).

## 5. Demos and source layout

Five pairs of Jittor-policy and scripted-expert videos are available in the [demo directory](reports/demos/five_task_pairs_20261031/README.md). Each pair uses the same initial scene. The learned-policy videos replay saved actions; the final second is a labelled freeze frame.

`src/quest_repro/` contains the models and shared data interface. `scripts/` contains collection, training, evaluation, and audit tools. `records/` stores experiment protocols; `reports/` stores results and lightweight logs. The [documentation index](docs/README.md) links the detailed guides.

## 6. References and license

- [QueST paper](https://arxiv.org/abs/2407.15840) and [official code](https://github.com/pairlab/QueST)
- [Jittor documentation](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
- [DMANet-Jittor](https://github.com/Yuyciciccc/DMANet-Jittor), referenced for the environment/data/training/results organization

This repository uses the [MIT License](LICENSE). Attribution, pinned upstream versions, and third-party licenses are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
