# 4090D 实际运行结果：2026-10-28

本轮工作已于北京时间约18:52正常完成。状态证据：
`gpu_verified_training_20261028/outputs/gpu_20261028_104533/workflow.json`。
服务器保持开机；本轮训练和性能测试已结束。

## 已完成的修复与检查

- Jittor Python 配置路径、CPU/GPU 缓存隔离、GPU 启用时机、Mish 接口问题已修复。
- Torch2.2 GPU 融合 Encoder 导致的一处 token 差异已通过 exact-GELU 普通层路径修复，详见 `GPU_TOKEN_FIX.md`。
- Jittor 并行算子编译偶发内存错误：关闭并行算子编译后，本轮 CPU/GPU 对齐、训练、计时均完成。
- 24项本地检查通过；tiny/full 分词器和策略模型的 CPU/GPU 前向、梯度检查通过。
- 全量11792个 token 整数完全一致，重建按原始 atol=2e-5、rtol=2e-4 通过。
- 20步预测/token/loss轨迹通过。**全部最终参数并非严格一致**：注意力 K bias 按解析不变性单独验证等价，原始严格失败报告保留，见 `JITTOR_RUNTIME_STATUS.md`。

## 配对训练

同一4090D、同一已训练分词器NPZ初始化、同一冻结数据划分、同一batch计划，dropout=0，各10轮、740次更新。属于配对微调，不是从零训练。

| 项目 | PyTorch | 原生Jittor |
|---|---:|---:|
| 最佳验证重建 MAE | 0.01066348 | 0.00952502 |
| GPU核心推理中位延迟，batch=16 | 7.297 ms | 7.864 ms |
| GPU核心推理P95，batch=16 | 8.610 ms | 8.108 ms |

两套训练曲线总体下降，但逐轮值并不完全相同；不能把这组单次结果解释为Jittor算法效果更优。延迟为encode→FSQ→decode，FP32，关闭TF32，预热10次、测量50次并同步，包含输出到主机；Torch Encoder使用上述普通层路径。它不是策略闭环控制频率。Jittor峰值显存未单独测量。

[训练曲线](gpu_verified_training_20261028/outputs/gpu_20261028_104533/paired_report_v2/paired_training_curves.png)

## 本机已校验的备份

- `gpu_verified_training_20261028.zip`，48386957字节；SHA256：
  `4edd6127e9b671f5ea3117259381cf8f2dae86b6de8105e5d1e776f0ac39042d`。
- SHA256、ZIP CRC和解压路径边界均已验证。
- 解压目录 `gpu_verified_training_20261028/` 包含两端真实日志、配置、曲线、对齐报告与最佳模型权重；完整优化器checkpoint仍在服务器。
- 新模型尚未用于原有条件策略。分词器变化后不能直接假定旧策略兼容，需重新训练或验证。

下一步仍需原生Jittor条件策略训练、开发集仿真评估、最后保留种子的测试，以及GitHub/PPT/视频交付。整体论文复现尚未完成。
