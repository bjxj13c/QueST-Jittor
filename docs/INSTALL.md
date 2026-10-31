## 环境配置

主要 GPU 实验使用以下环境：

| 项目 | 版本或配置 |
|---|---|
| 系统 / Python | Ubuntu 22.04 / Python 3.10.22 |
| GPU | RTX 4090D，24GB显存 |
| CUDA Toolkit / g++ | 12.1 / 11.4 |
| PyTorch / torchvision | 2.2.0＋cu121 / 0.17.0＋cu121 |
| Jittor | 1.3.11.0 |
| NumPy / h5py | 1.26.4 / 3.11.0 |
| MuJoCo / Gymnasium | 2.3.7 / 0.29.1 |
| MetaWorld | 固定提交 `c822f28f582ba1ad49eb5dcf61016566f28003ba` |

建议分开建立 PyTorch 和 Jittor 环境。脚本只创建项目内的 venv，需要系统已有 Python 3.10、g++ 和兼容的 CUDA 环境，不会安装显卡驱动。以下命令在仓库根目录执行。

```bash
bash scripts/bootstrap_linux.sh torch /path/to/python3.10 cu121
bash scripts/bootstrap_linux.sh jittor /path/to/python3.10 cpu

.venv-torch-linux/bin/python -m pip install -r requirements/simulation.txt -c requirements/cpu.txt
.venv-jittor-linux/bin/python -m pip install -r requirements/simulation.txt -c requirements/cpu.txt

.venv-torch-linux/bin/python scripts/linux_preflight.py --output reports/linux_preflight.json
.venv-jittor-linux/bin/python scripts/run_jittor_linux.py --device cuda scripts/jittor_cpu_probe.py --device cuda --output reports/jittor_cuda_probe.json
```

Jittor 安装参数中的 `cpu` 不限制后续使用 GPU；实际设备由 `run_jittor_linux.py --device cuda` 指定。该入口还隔离 CPU/CUDA 编译缓存，处理 Python 配置路径和并行算子编译问题。

无桌面服务器使用 EGL 渲染：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_TF32_OVERRIDE=0
```

若提示缺少 `libEGL.so.1`，需要准备系统 EGL 加载库，仅设置环境变量不能代替它。完整依赖见 [requirements/](../requirements/)；本地 CPU 试跑和 Linux 基础见[学习路线](LEARNING_ROUTE.md)。
