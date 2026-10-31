# Jittor 运行状态

截至2026-10-28，Linux Jittor1.3.11.0已完成CPU/GPU原生模型验证、10轮分词器配对微调、20轮条件策略训练和5个开发种子的真实仿真，原生执行未导入Torch。见 `GPU_RUN_RESULTS.md` 与 `POLICY_GPU_RESULTS.md`。最终测试、泛化改进和交付仍待完成，不能等同完整论文benchmark复现。

Windows 的独立 `.venv-jittor` 安装记录及失败仍保留；没有安装驱动、更改系统设置或修改原 Conda 环境。

Linux 4090D 首次后台任务于北京时间 12:46 在 GPU 初始化阶段停止，未进入训练。原始报告和日志为 `reports/gpu_4090d_first_workflow_stopped.json`、`reports/gpu_4090d_first_cuda_failure.log`。入口已分离 CPU/GPU 缓存，并将 use_cuda 的启用移到 import 完成以后。重试的 GPU 前向、梯度和保存/读取测试已实际通过，未导入 Torch；证据为 `reports/jittor_linux_cuda_probe_passed.json`。框架可运行不等于模型对齐通过。

北京时间 18:13，CPU 整条流程已通过。证据目录：`reports/alignment_cpu_20261028_verified/`（由实际服务器日志与报告下载而来）。tiny/full tokenizer 与 tiny/full policy 的前向、索引、loss、梯度通过原定容差；tokenizer 显式与原生 AdamW 一步更新通过；20 步预测/token/loss 轨迹通过；全部 1176 训练窗口和 298 验证窗口的 token 与重建比较通过。

**参数资格说明：20 步后的严格全部参数比较未通过。** 差异仅位于 decoder 第 0 层 self-attention 的 K bias，4 个元素超出阈值，最大差约 6.07e-5。原始失败报告 `trace_final_weights_comparison.json` 保留。按 softmax 对每行常量偏移的不变性，单独执行 K-bias 等价检查；对两套权重各增加 [-0.5,0.5] 偏置后，预测最大变化约 3.58e-7。其余参数维持原阈值。`trace_final_weights_equivalence.json` 明确记录 `strict_all_parameters_passed=false`，不能描述为全部权重严格一致。比较器的正/负检查记录在 `reports/key_bias_comparator_checks.json`。

GPU 全量 token 差异已修复，见 `reports/GPU_TOKEN_FIX.md`。根因定位为 Torch2.2 的融合版 Encoder eval 路径；通过 public exact-GELU callable 选择普通层计算路径后，全部11792个GPU token整数完全一致，重建按原阈值通过。证据已下载到 `reports/exactgelu_gpu_train_comparison.json` 和 `reports/exactgelu_gpu_validation_comparison.json`。

复查时Jittor并行算子编译线程偶发segfault；入口改用use_parallel_op_compiler=0。本轮18:45启动的完整流水线已于约18:52正常完成，CPU/CUDA门控、两端10轮微调和GPU计时均通过。两套框架均设置NVIDIA_TF32_OVERRIDE=0。两端最佳验证MAE为0.01066348与0.00952502；长训练曲线不是逐位一致，需按实际指标比较。

真实失败记录保留在本机 `reports/jittor_cpu_probe*.log`。使用已有 Visual Studio 2022 MSVC 19.44 时，Jittor 随包 `data.gz` 展开的 C++ 在成员函数指针转换处出现 `C2440`。排查时关闭 `use_data_gz` 后发现 PyPI 包没有提供全部核心 `.cc` 实现，最终出现 `Node` 等符号的链接错误；因此这条绕过方案不成立。当前入口已恢复保留 `data.gz`，不继续反复尝试同一编译。

原生动作分词器、FSQ、Transformer、部分ResNet18、SpatialSoftmax、BatchNorm和SkillGPT均经过CPU/GPU前向与梯度比较。策略两端20轮训练的best验证CE为1.323512与1.276619，开发成功2/5与1/5。训练后Jittor同一套权重的12个验证输入，在两端生成的96个token完全一致，动作误差最大5.36e-7。Torch不参与原生Jittor模型计算。

下一步在用户正常开启的 Linux 实例上，使用已有 Python 3.10 / g++ / CUDA 工具链：

1. `bash scripts/bootstrap_linux.sh jittor /path/to/python3.10 cpu`
2. `.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cpu scripts/jittor_cpu_probe.py`
3. 用同一 entry 执行 `run_jittor_fixture.py`，先验证 tiny 分词器的前向、索引、loss、输入梯度、参数梯度。
4. 用 `compare_npz.py` 比较真实候选文件，整数索引必须完全一致；误差阈值在命令和报告中保留，不偷偷放宽。
5. 再测试默认完整结构、策略网络、真实 AdamW 更新和短训练轨迹，最后才开展 GPU 性能比较。

`train_tokenizer_jittor.py`、`train_policy_jittor.py`、`evaluate_policy_jittor.py` 和 `benchmark_tokenizer.py` 的 Jittor 分支也是待 Linux 执行验证的入口。相同 seed 不代表不同框架使用相同初始化、dropout 和 batch 顺序；严格对齐需要 NPZ 权重及固定输入。Jittor 的 `.eval()` 会停止参数梯度，而 PyTorch 的 `.eval()` 不会，因此梯度对照使用专用 helper 保留参数梯度。完整流水线还会比较训练/验证全部 1474 个动作块的 token 与重建结果，而不只检查两个 fixture 样本。
