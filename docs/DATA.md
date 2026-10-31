## 数据准备

五任务基线每个任务90条训练、20条验证示范。示范由 MetaWorld 的脚本专家采集，参考作者的采集程序；不是训练好的专家神经网络。划分按 episode 和目标位置隔离，失败示范没有筛除。

| 数据 | 格式 |
|---|---|
| images | `[T, 128, 128, 3]`，uint8 RGB |
| robot_states | `[T, 8]` |
| actions | `[T, 4]` |
| success 等 | 每步执行后的反馈 |

图像和状态取自执行当前动作之前，策略内部将 RGB 除以255。动作块不跨 episode；末尾不足32步时重复最后一个动作，并记录有效位置 mask。

```text
outputs/five_task_10pct_20261029/
├── dataset/
│   ├── data/demonstrations.hdf5
│   ├── records/reach_split_v1.json
│   └── assets/
│       ├── weights.npz
│       ├── config.json
│       ├── shared/tokenizer_shared.pt
│       └── canonical_labels/{train,validation}.npz
└── experiments/
    ├── initial_five.npz
    ├── policy_torch/{best.pt,best_weights.npz,config.json}
    └── policy_jittor/{best_weights.npz,config.json}
```

数据和权重保存在实验归档中，**公开下载地址尚未补充**，只下载源码还不能直接测试已有模型。主数据 SHA256 为 `c899d5923933b57a8957fe20a48554756987f8ab72ae772f8b2adb1f3454fab9`。

采集与转换入口是 [collect_official_reach.py](../scripts/collect_official_reach.py) 和 [prepare_five_task_10pct.py](../scripts/prepare_five_task_10pct.py)。后者依赖前期三任务目录，不能在空目录里直接运行。[数据与运行说明](REPRODUCTION_GUIDE.md)列出了这些依赖。

若想先跑通接口，可以另建一个 Reach 小数据目录：

```bash
.venv-torch-linux/bin/python scripts/collect_reach.py --root data_runs/reach_demo --episodes 30
.venv-torch-linux/bin/python scripts/prepare_data.py --root data_runs/reach_demo
```

这组无噪声小数据用于试跑，不对应下表中的五任务成绩。作者代码通过 `scripts/fetch_upstream.py` 获取，固定在 `cc25a835a00cf2d91df69704bdd757904d13c061`。
