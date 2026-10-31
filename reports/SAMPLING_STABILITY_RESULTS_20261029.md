# 均衡采样：三随机种子稳定性检验

固定数据和分词器，两组配对初始化及训练预算；新评估种子16000–16029，每任务30局。

| 框架 | 训练种子 | 原采样成功/150 | 均衡采样成功/150 | 差值 |
|---|---:|---:|---:|---:|
| torch | 310500 | 107 | 104 | -2.00pp |
| torch | 310501 | 105 | 97 | -5.33pp |
| torch | 310502 | 95 | 91 | -2.67pp |
| jittor | 310500 | 103 | 102 | -0.67pp |
| jittor | 310501 | 100 | 104 | +2.67pp |
| jittor | 310502 | 97 | 105 | +5.33pp |

以训练种子作为统计单位，报告配对差值的均值、样本标准差及df=2的近似t区间。不能把1800次rollout当作1800次独立训练。
{
  "torch": {
    "mean_paired_difference": -0.033333333333333326,
    "seed_difference_sample_std": 0.017638342073763948,
    "paired_seed_t95_interval": [
      -0.07714940405305587,
      0.010482737386389225
    ],
    "positive_runs": 0,
    "mean_success_rates": {
      "uniform": 0.6822222222222223,
      "balanced": 0.6488888888888888
    },
    "interpretation": "not enough evidence of stable improvement across training seeds"
  },
  "jittor": {
    "mean_paired_difference": 0.02444444444444449,
    "seed_difference_sample_std": 0.030061665018819267,
    "paired_seed_t95_interval": [
      -0.0502328713094229,
      0.09912176019831188
    ],
    "positive_runs": 2,
    "mean_success_rates": {
      "uniform": 0.6666666666666666,
      "balanced": 0.6911111111111111
    },
    "interpretation": "not enough evidence of stable improvement across training seeds"
  }
}