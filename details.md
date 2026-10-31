# QueST框架迁移与对齐细节

本文件说明两阶段模型的原生Jittor实现，以及与PyTorch参考版本的对照方法。训练命令见[训练说明](docs/TRAINING.md)，结果与扩展实验见[README](README.md)及[实验文档](docs/EXPERIMENTS.md)。

## 一、实现结构

| 模块 | Jittor实现 | PyTorch参考 |
|---|---|---|
| 动作分词器、FSQ与注意力 | [tokenizer_jittor.py](src/quest_repro/tokenizer_jittor.py) | [tokenizer_torch.py](src/quest_repro/tokenizer_torch.py) |
| 图像编码与条件策略 | [policy_jittor.py](src/quest_repro/policy_jittor.py) | [policy_torch.py](src/quest_repro/policy_torch.py) |
| 轨迹切片与末尾补齐 | [data.py](src/quest_repro/data.py) | 两框架共用NumPy接口 |
| token标签缓存 | [policy_data_jittor.py](src/quest_repro/policy_data_jittor.py) | [policy_data.py](src/quest_repro/policy_data.py) |
| 参数分组与导出 | [jittor_training.py](src/quest_repro/jittor_training.py) | [training.py](src/quest_repro/training.py) |

原生Jittor模型的训练、推理和参数更新不执行PyTorch模型。跨框架共享权重通过独立进程导出静态NPZ数组完成。

## 二、输入与动作表示

策略接收原始uint8图像，在模型入口除以255；机器人状态为8维，任务由ID指定。39维完整环境观察用于脚本专家和数据检查，不作为学习策略输入。

动作片段长32步，每步4维。时间卷积将序列缩短为8个位置，Transformer编码后投影到4维量化空间。FSQ各维档位为`[8,5,5,5]`，对应1000种编号，一段动作生成8个token。解码器将量化特征重建为32步动作。

动作片段不跨轨迹。末尾不足32步时重复最后一个动作，并用`pad_mask`标记真实位置。主基线的训练损失包含补齐位置；有效位置MAE另外记录，供分析使用。

## 三、FSQ与因果注意力

FSQ依次执行投影、限幅、直通取整和编号转换。Jittor中显式实现了与PyTorch一致的半整数取整规则；直通估计使梯度能够通过取整操作传回编码器。固定的是量化网格，编码器、解码器与投影层继续训练。

注意力实现分别计算Q、K、V，再按多头拆分。因果掩码将未来位置设为负无穷，经softmax后对应权重为0。迁移时逐项检查拆分、转置和softmax维度，并对照中间特征。

核心文件已补充中文注释。注释整理前后经过AST比较，原有模型表达式和训练配置保持一致。

## 四、条件输入与训练

图像编码为256维，8维状态编码为128维，拼接后投影为384维观察特征。任务ID也映射为384维向量，放在观察特征前；单帧配置的条件序列形状为`[B,2,384]`。

策略训练采用错开一位的token输入与目标，在因果掩码下并行计算交叉熵。实际执行时逐个生成8个token，由冻结分词器解码，再执行前8步动作并重新观察。

两框架配对策略共享同组冻结分词器和token标签。参数初始化以相同NPZ为准，训练顺序由共同批次计划或固定随机流控制。训练记录包含验证交叉熵、裁剪前梯度范数和样本顺序哈希。

## 五、归一化与参数状态

LayerNorm、GroupNorm和BatchNorm中的统计方式、epsilon、参数及运行状态均需要核对。Jittor策略的BatchNorm保留PyTorch更新运行方差时采用的无偏估计；均值和方差计算在模型文件中显式实现。

另外处理了不参与实际前向计算的注册参数，使优化器的参数更新行为与参考实现对应。权重加载会检查名称、形状和类型，避免静默忽略参数。

## 六、数值对齐

检查使用相同模型权重、输入与采样方式。五任务已训练模型的200组验证输入得到：

| 项目 | 结果 |
|---|---:|
| 条件特征最大绝对误差 | 1.1772e-6 |
| 1600个动作token | 全部一致 |
| 解码动作最大绝对误差 | 1.8418e-5 |

浮点使用联合标准`abs(error) <= 2e-5 + 2e-4 * abs(reference)`，整数token要求精确相同。该组前向检查通过。参考文件与独立报告见[五任务结果](reports/FIVE_TASK_10PCT_RESULTS_20261029.md)。

前向、梯度、一步更新和短程训练的检查入口为`scripts/verify_jittor_alignment.sh`，所需fixture及生成方法见[运行补充说明](docs/REPRODUCTION_GUIDE.md#5-对齐检查的准备和解释)。

历史短程参数比较中出现过注意力K-bias的严格比较失败，原记录及解析等价性解释保存在[Jittor运行记录](reports/JITTOR_RUNTIME_STATUS.md)。这部分与已训练模型的前向对齐结果分别记录。

## 七、保持实验中的数值修复

Mish＋GELU保持分支曾出现token相同、少量解码动作超过联合容差的情况。逐层对照观察到末端归一化附近误差放大，随后在独立补测分支中，将冻结Jittor解码器的12个LayerNorm内部计算改为FP64。

参数、输入输出仍为FP32，权重与归一化公式不变，没有重新训练。复查中1600个token全部一致，动作最大绝对误差为2.39089e-5，按原绝对与相对误差联合标准通过，随后补齐360次评估。

修复仅用于这一保持补测分支；原实现的失败记录保留，三种子独立确认使用原实现。设置和证据见[补测记录](reports/HOLDING_COMPLETION_20261030.md)、[修复协议](records/decoder_precision_patch_20261030.json)与[独立核验](reports/HOLDING_COMPLETION_LOCAL_VERIFICATION_20261030.json)。

## 八、结果复查

仓库保留实验协议、汇总JSON、CSV曲线和轻量日志。完整数据、权重与逐场景数组需从实验归档恢复，文件对应关系见[数据说明](docs/DATA.md)。

```bash
python scripts/verify_activation_confirmation.py \
  --evidence-root reports/five_task_evidence_20261029 \
  --output reports/confirmation_check.json
```

该命令读取保存的模型与实验记录，检查预算、初始化、批次、场景和成功率统计。1800次是累计任务测试次数，每个框架的训练稳定性仍按三个种子的配对差值分析。

文档组织参考[DMANet-Jittor](https://github.com/Yuyciciccc/DMANet-Jittor)。模型实现与许可证来源见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
