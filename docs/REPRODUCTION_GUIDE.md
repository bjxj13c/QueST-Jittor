# 数据与运行补充说明

README 给出主要入口，这里补充归档依赖、历史目录和对齐检查。主基线是五任务450条训练、100条验证，扩展实验另外记录，不混成一组结果。

## 1. 先确认使用哪条路线

| 目的 | 所需材料 | 对应结果 |
|---|---|---|
| 先跑通数据接口 | 按 README 重新采集30条 Reach | 新的小数据试跑，不对应已报告成绩 |
| 复查五任务基线 | 主数据、公共旧初始化、已训练模型或重新训练 | PyTorch 72%、Jittor 68% 那轮 |
| 检查12组激活函数 | 五任务数据，激活实验脚本和协议 | 新初始化、新场景下的完整4×3对照 |
| 优选组合＋保持示范 | 激活实验检查点、配对Reach数据、其他四任务数据 | 第二轮配对微调，当前7/8单元完成评估 |

公开数据和权重下载地址尚未填写，不能把已有服务器目录的运行流程写成“全新 clone 一键复现”。重新采集会生成新的数据和哈希，也不应声称与已保存数据逐字节相同。

## 2. 五任务数据怎样构建

采集参考 QueST 官方 `scripts/generate_metaworld_dataset.py`，本仓库用 `scripts/collect_official_reach.py --tasks ...` 限定任务、示范数和随机种子。固定作者代码提交为 `cc25a835a00cf2d91df69704bdd757904d13c061`。

`scripts/prepare_five_task_10pct.py` 会：

1. 读取前期三任务数据 `outputs/overnight_20261029/datasets/multi240`，保留已有训练/验证成员。
2. 加入 Door-open、Window-open 的示范，以及每个任务新增的10条训练示范。
3. 将作者数据中 T+1 个观察与 T 个动作对应起来，保留执行动作之前的图像、状态。
4. 检查图像、动作、状态长度和范围，按目标位置隔离训练与验证。
5. 生成五任务 HDF5、划分清单和仅使用训练集计算的统计信息。统计信息保留用于检查，没有据此重新缩放控制量。

完整设置见 [five_task_10pct_20261029.json](../records/five_task_10pct_20261029.json)。该构建脚本依赖已有三任务目录，不是通用的任意数据集转换器。

作者采集器还需要 upstream 环境中的 Hydra、OmegaConf、robomimic 等依赖，不能只安装本项目的最小数据检查依赖就假定可运行。实际采集环境另行隔离；主训练和推理使用 README 中的环境。数据准备和安装入口分别为：

- `scripts/fetch_upstream.py`：获取固定提交的作者参考代码。
- `scripts/collect_official_reach.py`：运行作者采集器。
- `scripts/validate_official_reach.py`：单任务作者数据检查和转换。
- `scripts/prepare_five_task_10pct.py`：本次五任务数据合并和划分。
- `scripts/collect_reach.py`、`scripts/prepare_data.py`：较简单的独立 Reach 试跑路线。

## 3. 复查原五任务成绩需要哪些初始化

README 中的分步训练不是从零训练，依赖下面两份初始化：

| 文件 | 来源 |
|---|---|
| `outputs/gpu_20261028_104533_tokenizer_jittor/best_weights.npz` | 前期30条 Reach 训练得到的原生分词器，随后在450条训练示范上适配10轮 |
| `outputs/five_task_10pct_20261029/experiments/initial_five.npz` | 公共策略初始化；保留原 task 0 的嵌入，并按固定种子补充另外四行 |

原来的策略公共初始化来自 `outputs/fixtures/policy_full_seed456/weights.npz`，新增 task 行的种子为310456。两端实际读取同一份 NPZ，而不是各自设一个相同随机种子就认为权重一致。

若改成随机初始化分词器，就成为另一组实验。12组激活对照正是使用全新公共初始化，因此它的原始激活基线成绩与这轮72%/68%不直接可比。

原始流程保存在 `scripts/run_five_task_10pct.py` 和实验目录的 `workflow.json` 中。这个总调度器会检查已完成状态并拒绝重复运行；在已有完成目录里复查时，先读记录，不要直接重启它。README 的 `readme_*` 名称用于另建输出，重复执行同一名称也会被拒绝，需要换新名称。

## 4. 只做一个从零的小数据试跑

先按 README 采集和准备 `data_runs/reach_demo`。随后可以用下列入口导出一份公共随机初始化：

```bash
.venv-torch-linux/bin/python scripts/export_tokenizer_fixture.py \
  --root data_runs/reach_demo --arch full --name tokenizer_initial --seed 123
```

输出在 `data_runs/reach_demo/outputs/fixtures/tokenizer_initial/`，其中 `weights.npz` 是更新之前的初始权重；`updated_weights.npz` 是检查用的一步更新结果，不能把二者混用。将该 `weights.npz` 作为两端分词器的 `--initial-weights`，按新数据创建自己的 batch plan，再训练、冻结和生成标签。

单任务策略也应使用单任务初始化，不能直接加载五行 task embedding 的 `initial_five.npz`。可在完成分词器静态容器转换后，用 `scripts/export_policy_fixture.py --arch full --tokenizer ...` 导出单任务公共初始权重。该路线用于学习和检查接口，尚未作为全新环境下的独立端到端交付验收；不能用主结果表替代它的新实验结果。

## 5. 对齐检查的准备和解释

`scripts/verify_jittor_alignment.sh cuda` 需要两套环境及以下参考目录：

```text
outputs/fixtures/tokenizer_tiny_seed123/
outputs/fixtures/tokenizer_full_trained/
outputs/fixtures/policy_tiny_seed456/
outputs/fixtures/policy_full_seed456/
```

生成入口为 `scripts/export_tokenizer_fixture.py` 和 `scripts/export_policy_fixture.py`；全尺寸已训练分词器参考要用匹配的 `--checkpoint`。每个目录记录实际配置、输入、权重和参考输出。只有依赖准备齐全后才运行完整对齐脚本，它不会自动替你补出缺失的训练模型。

这个批量检查脚本固定读取项目根目录的旧数据与 fixture，并不接受前面小数据路线的 `--root`。对独立数据根目录，需恢复完整旧参考，或分别运行导出、Jittor fixture 和 NPZ 比较入口，不能直接把子目录的参考当成已准备好的默认目录。

```bash
bash scripts/verify_jittor_alignment.sh cuda
```

该脚本检查前向、梯度、一步更新、短程训练轨迹及 token 一致性。注意力 K-bias 的历史严格参数比较失败保留原报告，并单独说明解析等价性；没有对其他参数放宽阈值。长期训练中，两端浮点运算和优化轨迹会逐步产生差异，应该同时查看实现对齐、训练曲线和在线评估。

五任务已训练模型另用 `scripts/export_trained_policy_predictions.py` 在相同权重下生成条件向量、token 和连续动作参考，报告入口是 `scripts/audit_five_task_10pct.py --phase report`。复查旧测试种子是核验原结果，不应再用这些结果调参后称为新测试。

## 6. 激活函数对照与测速

激活函数由 `scripts/run_activation_variant.py` 在独立进程中选择，不改共享模型源码。训练、标签导出、对齐、测试和测速都需要经过同一入口；直接用默认模型类加载 SiLU/ReLU 变体会使用错误的激活。

完整矩阵的设置保存在 [activation_grid_protocol_20261029.json](../records/activation_grid_protocol_20261029.json)，调度入口为 `scripts/run_activation_grid.py`。它依赖现有五任务数据；已有完成目录不能直接重复运行。

以已完成的 Mish＋GELU 为例，独占测速命令如下。输出已存在时脚本会拒绝覆盖；查现有成绩时直接读对应 JSON 即可。

```bash
.venv-torch-linux/bin/python scripts/run_activation_variant.py \
  --framework torch --conv mish --ffn gelu scripts/benchmark_activation_case.py \
  --framework torch --case-root outputs/activation_grid_20261029/cases/mish_gelu

.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cuda \
  scripts/run_activation_variant.py --framework jittor --conv mish --ffn gelu \
  scripts/benchmark_activation_case.py --framework jittor \
  --case-root outputs/activation_grid_20261029/cases/mish_gelu
```

测速在矩阵训练结束后逐组单独进行，固定预热、同步和测量次数。并发训练的总耗时也保存，但受其他进程影响，不能与独占计时混为一张加速比表。

## 7. 保持示范与第二轮数据

Reach 的配对采集保存了相同的到达前轨迹，一支在首次成功时结束，另一支由专家继续控制60步。没有人为填零控制量。随后比较重复补齐、屏蔽重建补齐损失和真实保持示范；仅屏蔽重建损失没有去掉编码器输入或策略标签里的补齐动作。

后续优选组合实验保留其他四任务，将 Reach 替换成配对数据。为避免重复存储，第二轮的 HDF5 使用 external links 引用原五任务与配对 Reach 文件，所以不能只复制一个链接容器。需要按项目相对目录恢复完整归档，源文件 SHA 也在实验清单中核对。

初次 Reach 对照的分词器2,000次更新、策略6,000次更新；优选组合配对微调为分词器2,000次、策略4,000次。二者不能混为同一次实验。第二轮的原始激活保持分支已在新 RTX4090 实例补齐360次评估，见[补测说明](../reports/HOLDING_COMPLETION_20261030.md)。原FP32检查失败保留；补测通过的是独立的数值实现修复分支，未重新训练。

补测入口为 `scripts/complete_missing_holding_precision.py`，通过 `scripts/run_stable_decoder.py` 在当前进程中替换冻结 Jittor 解码器的12个 LayerNorm：内部 FP64 累加，参数与接口 FP32。使用原检查点与固定场景，先通过原 `atol=2e-5, rtol=2e-4` 联合检查和完全相同 token，再评估。它不是默认原FP32实现的无修改复跑；旧实验和源代码不覆盖。归档下载后可在本机CPU独立复核：

```powershell
.venv-torch/Scripts/python.exe scripts/verify_holding_completion.py --archive reports/holding_completion_precision_20261030.zip --workflow reports/HOLDING_COMPLETION_WORKFLOW_20261030.json
```

该命令只核验已有记录、哈希与数组，不训练或重新仿真。归档中间状态与另存的最终 workflow 分开保存，以最终 workflow 的完成状态和归档SHA为准。

## 8. 文档与发布材料

README 结构参考了 Jittor-Sprouts 收录项目，主要借鉴环境、数据、运行命令、对齐结果分开说明的写法：

- [DMANet-Jittor](https://github.com/Yuyciciccc/DMANet-Jittor)：环境、数据子集、训练测试和两框架结果分别说明。
- [LIIF-Jittor](https://github.com/WindATree/LIIF-Jittor)：配置、数据、实现细节和对应命令集中列出。
- [RepVGG-Jittor-CIFAR100](https://github.com/NeikuiColacat/RepVGG-Jittor-CIFAR100)：提供相互对应的 PyTorch/Jittor 入口。

这里只参考组织方式，不沿用其数据、成绩或安装修复方案。大文件发布地址、个人仓库地址及干净环境复跑检查仍待本人验收。旧 README 已在本机 `backups/readme_revision_20261030/` 留档。
