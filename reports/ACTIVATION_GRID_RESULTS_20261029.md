# QueST激活函数完整4×3对照

| 卷积 | FFN | 框架 | 成功/150 | 重建MAE(有效位置) | 策略最佳步 |
|---|---|---|---:|---:|---:|
| mish | gelu | torch | 91 | 0.131341 | 7000 |
| mish | gelu | jittor | 95 | 0.131125 | 7000 |
| mish | silu | torch | 96 | 0.133918 | 8000 |
| mish | silu | jittor | 92 | 0.133240 | 8000 |
| mish | relu | torch | 101 | 0.127769 | 8000 |
| mish | relu | jittor | 102 | 0.128437 | 8000 |
| silu | gelu | torch | 94 | 0.130884 | 6000 |
| silu | gelu | jittor | 99 | 0.130859 | 9000 |
| silu | silu | torch | 99 | 0.131865 | 7000 |
| silu | silu | jittor | 90 | 0.132003 | 6000 |
| silu | relu | torch | 105 | 0.128463 | 8000 |
| silu | relu | jittor | 95 | 0.128751 | 8000 |
| gelu | gelu | torch | 92 | 0.129854 | 7000 |
| gelu | gelu | jittor | 96 | 0.130075 | 7000 |
| gelu | silu | torch | 99 | 0.132475 | 8000 |
| gelu | silu | jittor | 88 | 0.132195 | 6000 |
| gelu | relu | torch | 101 | 0.128536 | 8000 |
| gelu | relu | jittor | 100 | 0.128542 | 8000 |
| relu | gelu | torch | 100 | 0.130422 | 8000 |
| relu | gelu | jittor | 96 | 0.130558 | 8000 |
| relu | silu | torch | 90 | 0.132750 | 8000 |
| relu | silu | jittor | 98 | 0.132715 | 8000 |
| relu | relu | torch | 97 | 0.128429 | 8000 |
| relu | relu | jittor | 94 | 0.128656 | 8000 |

所有失败组保留日志。单训练种子，不能把最优组合直接称为稳定提升。收敛曲线、分任务结果和耗时边界见JSON。
ResNet的ReLU及FSQ有界变换不变，Transformer激活同时用于分词器编码器/解码器与条件策略。