## 训练

下面分步运行五任务基线，使用新输出目录，不覆盖原实验。开始前需要恢复数据、公共分词器初始化和策略初始化，来源见[运行说明](REPRODUCTION_GUIDE.md)。主结果的分词器从已有权重适配10轮，不是从零训练。

### 1. 动作分词器

```bash
QUEST_DATA=outputs/five_task_10pct_20261029/dataset
QUEST_TOKEN_INIT=outputs/gpu_20261028_104533_tokenizer_jittor/best_weights.npz
QUEST_PLAN="$QUEST_DATA/records/tokenizer_plan_readme.npz"
QUEST_JT=(.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cuda)

.venv-torch-linux/bin/python scripts/create_batch_plan.py \
  --root "$QUEST_DATA" --output "$QUEST_PLAN" --epochs 10 --batch-size 16 --seed 310300

.venv-torch-linux/bin/python train_pytorch.py --stage tokenizer \
  --root "$QUEST_DATA" --name readme_tokenizer_torch --device cuda \
  --epochs 10 --batch-size 16 --dropout 0 --seed 310300 \
  --initial-weights "$QUEST_TOKEN_INIT" --batch-plan "$QUEST_PLAN"

.venv-jittor-linux/bin/python train_jittor.py --stage tokenizer \
  --root "$QUEST_DATA" --name readme_tokenizer_jittor --device cuda \
  --epochs 10 --batch-size 16 --dropout 0 --seed 310300 \
  --initial-weights "$QUEST_TOKEN_INIT" --batch-plan "$QUEST_PLAN"
```

两端读取相同权重和批次计划，优化全位置 L1 重建损失，同时记录包含补齐位置的 MAE 和仅真实位置的 MAE。

### 2. 冻结分词器、生成标签

两端条件策略使用同一个原生 Jittor 分词器检查点，避免分词器差异影响策略框架对照。PyTorch 容器转换只准备静态权重。

```bash
QUEST_ASSETS="$QUEST_DATA/readme_assets"
mkdir -p "$QUEST_ASSETS"
cp "$QUEST_DATA/outputs/readme_tokenizer_jittor/best_weights.npz" "$QUEST_ASSETS/weights.npz"
cp "$QUEST_DATA/outputs/readme_tokenizer_jittor/config.json" "$QUEST_ASSETS/config.json"

.venv-torch-linux/bin/python scripts/prepare_policy_stage.py \
  --root "$QUEST_DATA" --framework torch --prepare-checkpoint \
  --tokenizer-weights "$QUEST_ASSETS/weights.npz" --tokenizer-config "$QUEST_ASSETS/config.json" \
  --output "$QUEST_ASSETS/shared"

"${QUEST_JT[@]}" scripts/prepare_policy_stage.py \
  --root "$QUEST_DATA" --framework jittor \
  --tokenizer-weights "$QUEST_ASSETS/weights.npz" --tokenizer-config "$QUEST_ASSETS/config.json" \
  --output "$QUEST_ASSETS/canonical_labels"
```

### 3. 条件策略

```bash
QUEST_POLICY_INIT=outputs/five_task_10pct_20261029/experiments/initial_five.npz

for QUEST_FRAMEWORK in torch jittor; do
  if [ "$QUEST_FRAMEWORK" = torch ]; then
    QUEST_PY=(.venv-torch-linux/bin/python)
    QUEST_EXTRA=(--tokenizer-checkpoint "$QUEST_ASSETS/shared/tokenizer_shared.pt")
  else
    QUEST_PY=("${QUEST_JT[@]}")
    QUEST_EXTRA=()
  fi
  "${QUEST_PY[@]}" scripts/train_controlled_policy.py \
    --root "$QUEST_DATA" --output "$QUEST_DATA/outputs/readme_policy_$QUEST_FRAMEWORK" \
    --framework "$QUEST_FRAMEWORK" --tokenizer-weights "$QUEST_ASSETS/weights.npz" \
    --tokenizer-config "$QUEST_ASSETS/config.json" --initial-weights "$QUEST_POLICY_INIT" \
    --canonical-labels "$QUEST_ASSETS/canonical_labels" \
    --updates 12000 --eval-every 1000 --lr 0.0001 --weight-decay 0.0001 \
    --dropout 0 --shift 0 --seed 310500 "${QUEST_EXTRA[@]}"
done
```

每个 batch 为16个动作块，学习率余弦下降到 `1e-5`。分词器固定，策略按验证集交叉熵选择检查点。每步梯度范数保存在日志中，非有限 loss 或梯度会触发停止。


## 根目录入口

`train_jittor.py --stage tokenizer` 和 `--stage policy` 分别调用原生分词器训练与条件策略训练；`train_pytorch.py` 提供相同的两阶段 PyTorch 入口。模型、优化器和数据处理仍由原脚本执行。

在原生 Jittor 环境中训练策略，可用下面的命令替代上文循环的 Jittor 分支。运行前先完成分词器训练与标签生成。

```bash
.venv-jittor-linux/bin/python train_jittor.py --stage policy --device cuda \
  --root "$QUEST_DATA" --output "$QUEST_DATA/outputs/readme_policy_jittor" \
  --tokenizer-weights "$QUEST_ASSETS/weights.npz" \
  --tokenizer-config "$QUEST_ASSETS/config.json" \
  --initial-weights "$QUEST_POLICY_INIT" \
  --canonical-labels "$QUEST_ASSETS/canonical_labels" \
  --updates 12000 --eval-every 1000 --lr 0.0001 --weight-decay 0.0001 \
  --dropout 0 --shift 0 --seed 310500
```

使用 `--dry-run` 可先查看最终调用命令；该选项不会开始训练。条件策略训练入口使用 CUDA，分词器还支持 CPU 小规模检查。完整更新预算、随机种子和实际记录见 [实验协议](../records/five_task_10pct_20261029.json)。
