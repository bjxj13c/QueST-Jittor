## 测试

下面测试归档中的原五任务模型，先以开门任务为例。测试新训练模型时，将策略目录与分词器资产一起换成对应的 `readme_policy_*`、`readme_assets` 路径。

```bash
QUEST_DATA=outputs/five_task_10pct_20261029/dataset
QUEST_RUN=outputs/five_task_10pct_20261029/experiments
QUEST_ASSETS="$QUEST_DATA/assets"
QUEST_JT=(.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cuda)
QUEST_TASK=door-open-v2
QUEST_TASK_ID=3

.venv-torch-linux/bin/python scripts/evaluate_policy.py \
  --root "$QUEST_DATA" --policy "$QUEST_RUN/policy_torch/best.pt" \
  --tokenizer "$QUEST_ASSETS/shared/tokenizer_shared.pt" --sampling-backend numpy \
  --device cuda --seed-set experiment --seed-start 12000 --episodes 30 \
  --env-name "$QUEST_TASK-goal-observable" --task-id "$QUEST_TASK_ID" \
  --per-episode-sampling --top-k 5 --sampling-seed 30000 --max-steps 150 \
  --action-horizon 8 --videos 0 --name "readme_test_torch_$QUEST_TASK"

.venv-jittor-linux/bin/python test_jittor.py \
  --root "$QUEST_DATA" --policy-weights "$QUEST_RUN/policy_jittor/best_weights.npz" \
  --policy-config "$QUEST_RUN/policy_jittor/config.json" \
  --tokenizer-weights "$QUEST_ASSETS/weights.npz" --tokenizer-config "$QUEST_ASSETS/config.json" \
  --device cuda --seed-set experiment --seed-start 12000 --episodes 30 \
  --env-name "$QUEST_TASK-goal-observable" --task-id "$QUEST_TASK_ID" \
  --per-episode-sampling --top-k 5 --sampling-seed 30000 --max-steps 150 \
  --action-horizon 8 --videos 0 --name "readme_test_jittor_$QUEST_TASK"
```

五任务的 `QUEST_TASK / QUEST_TASK_ID` 依次为：`reach-v2 / 0`、`reach-wall-v2 / 1`、`push-v2 / 2`、`door-open-v2 / 3`、`window-open-v2 / 4`。

每个任务30个场景，最多150步，以环境返回的成功标志计算成功率。12000–12029为本轮已使用的测试种子，继续调参需要另留新场景。输出位于数据根目录的 `outputs/`，包括 `summary.json`、`episodes.jsonl`、逐场景动作和 token 数组。
