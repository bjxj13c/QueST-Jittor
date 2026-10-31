# 历史实验索引

早期实验保留用于说明实现过程，不与 README 中的五任务主结果合并统计。旧报告中的“下一步”反映当时状态，当前进度以 README 为准。

| 阶段 | 主要内容 | 记录 |
|---|---|---|
| 30条 Reach | 采集、动作块、分词器、第一版策略与CPU/GPU迁移 | [GPU运行记录](../reports/GPU_RUN_RESULTS.md)、[策略结果](../reports/POLICY_GPU_RESULTS.md) |
| 100条作者采集数据 | 目标隔离后的80/20划分、两阶段训练和20场景评估 | [正式结果](../reports/OFFICIAL100_FINAL_RESULTS.md)、[训练记录](../reports/OFFICIAL100_POLICY_RESULTS.md) |
| 扩数据和调参 | 80/280条训练数据、固定预算2×2对照 | [夜间实验](../reports/OVERNIGHT_RESULTS_20261029.md) |
| 三任务共享策略 | Reach、Reach-wall、Push联合训练 | [同一报告的三任务部分](../reports/OVERNIGHT_RESULTS_20261029.md) |
| 五任务450条训练示范 | 当前主基线，分词器适配与共享策略 | [五任务结果](../reports/FIVE_TASK_10PCT_RESULTS_20261029.md) |
| 反馈频率 | 固定模型，仅改变每次执行4/8/16步 | [结果](../reports/FEEDBACK_HORIZON_RESULTS_20261029.md) |
| 任务均衡采样 | 单次观察与三种子重复检验 | [单次](../reports/TASK_BALANCED_RESULTS_20261029.md)、[三种子](../reports/SAMPLING_STABILITY_RESULTS_20261029.md) |
| 视觉初始化 | 同样归一化下的随机/预训练初始化 | [结果](../reports/PRETRAINED_VISION_RESULTS_20261029.md) |
| 补齐与保持 | Reach单任务三种设置、后续优选激活组合配对微调 | [汇总](../reports/EXPERIMENT_CLOSEOUT_20261030.md)、[第二轮](../reports/SELECTED_HOLDING_RESULTS_20261030.md) |
| 激活函数矩阵 | 时间卷积4种×Transformer FFN 3种 | [12组结果](../reports/ACTIVATION_GRID_RESULTS_20261029.md) |

不同轮次有不同数据、初始化、任务数和测试种子。只有同一协议内的配对结果适合直接比较；单看两个历史成功率的差值，不能推出某一项改动有效。
