"""Derive a concise public report from preserved raw evidence, including failures."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
reports.mkdir(exist_ok=True)
figures = reports / "figures"
figures.mkdir(exist_ok=True)
sources = {}


def read(relative):
    path = root / relative
    if not path.exists():
        return None
    payload = path.read_bytes()
    sources[relative] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def wilson(successes, count):
    p, z = successes / count, 1.959963984540054
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return center-half, center+half


tokenizer = read("outputs/tokenizer_cpu_full_10epochs_v1/summary.json")
validation = read("outputs/tokenizer_cpu_full_10epochs_v1/evaluation_best/metrics.json")
policy = read("outputs/policy_cpu_full_5epochs_v1/summary.json")
warm = read("outputs/policy_cpu_warmstart5_plus10_v1/summary.json")
evaluations = {}
for name in ("policy_cpu_dev5_v1", "policy_cpu_dev5_zero_both_v1",
             "policy_cpu_warmstart_dev5_v1", "policy_cpu_warmstart_dev5_zero_both_v1"):
    data = read(f"outputs/{name}/summary.json")
    if data:
        evaluations[name] = data
for name, source in {
    "tokenizer_training.png": "outputs/tokenizer_cpu_full_10epochs_v1/training_curves.png",
    "tokenizer_reconstruction.png": "outputs/tokenizer_cpu_full_10epochs_v1/evaluation_best/validation_reconstructions.png",
    "policy_training.png": "outputs/policy_cpu_full_5epochs_v1/training_curves.png",
    "policy_additional_training.png": "outputs/policy_cpu_warmstart5_plus10_v1/training_curves.png",
}.items():
    path = root / source
    if path.exists():
        shutil.copy2(path, figures / name)
raw_copies = []
run_names = ["tokenizer_cpu_full_10epochs_v1", "policy_cpu_full_5epochs_v1", "policy_cpu_warmstart5_plus10_v1", *evaluations]
for run_name in run_names:
    for filename in ("config.json", "summary.json", "metrics.csv", "metrics.jsonl", "initial_validation.json",
                     "best_validation.json", "episodes.jsonl"):
        original = root / "outputs" / run_name / filename
        if original.exists():
            destination = reports / "runs" / run_name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            raw_copies.append({"source": str(original.relative_to(root)), "copy": str(destination.relative_to(root)),
                               "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
for filename, source in {
    "policy_success_dev2000.mp4": "outputs/policy_cpu_warmstart_dev5_v1/seed_2000.mp4",
    "policy_failure_dev2001.mp4": "outputs/policy_cpu_dev5_v1/seed_2001.mp4",
}.items():
    original = root / source
    if original.exists():
        destination = reports / "demos" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        raw_copies.append({"source": source, "copy": str(destination.relative_to(root)),
                           "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
(reports / "RUNS_MANIFEST.json").write_text(json.dumps(raw_copies, indent=2), encoding="utf-8")
final_runs = []
for path in (root / "outputs").glob("*/summary.json"):
    candidate = json.loads(path.read_text(encoding="utf-8"))
    if candidate.get("seed_set") == "final":
        final_runs.append(path.parent.name)
summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "tokenizer": tokenizer,
           "tokenizer_validation": validation, "policy": policy, "additional_training": warm,
           "development_evaluations": evaluations, "evidence_sha256": sources,
           "jittor": "runtime unverified; Windows compiler failure retained", "final_test_seeds_used": bool(final_runs),
           "final_runs_detected": final_runs}
(reports / "RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
lines = ["# 已执行结果与边界", "", "本表由 `scripts/generate_results.py` 从原始 JSON 生成，源文件哈希见 `RESULTS.json`。训练输出与早期失败没有被替换。", "",
         "## PyTorch 分词器", ""]
if tokenizer and validation:
    metrics = validation["metrics"]
    lines += [f"默认完整结构，CPU 4 线程，10 轮、{tokenizer['steps']} 次更新。最佳验证 MAE **{metrics['mae_all']:.6f}**；仅真实动作位置 MAE **{metrics['mae_valid']:.6f}**。",
              f"零动作基线 {metrics['zero_action_mae']:.6f}，训练均值动作基线 {metrics['train_mean_action_mae']:.6f}。", "",
              "![分词器训练曲线](figures/tokenizer_training.png)", "", "![验证集动作重建](figures/tokenizer_reconstruction.png)", ""]
lines += ["## PyTorch 条件策略", ""]
if policy:
    final = policy["final"]
    lines += [f"首版 5 轮的验证 CE {final['validation_ce']:.6f}，自回归动作 MAE {final['autoregressive_action_mae']:.6f}。", "",
              "![首版策略训练](figures/policy_training.png)", ""]
if warm:
    lines += [f"从首版最佳权重追加 10 轮，新建 AdamW 和 cosine schedule。最佳验证 CE {warm['best_validation_ce']:.6f}；末轮 CE {warm['final']['validation_ce']:.6f}。这是 warm-start 实验，不是精确断点恢复。", "",
              "![追加训练](figures/policy_additional_training.png)", ""]
lines += ["## 真正闭环的开发期结果", "", "| 设置 | 成功/回合 | 成功率 | Wilson 95% 区间 |", "|---|---:|---:|---:|"]
for name, result in evaluations.items():
    count = len(result["episodes"])
    successes = sum(bool(row["success"]) for row in result["episodes"])
    low, high = wilson(successes, count)
    lines.append(f"| {name} | {successes}/{count} | {successes/count:.0%} | [{low:.1%}, {high:.1%}] |")
lines += ["", "每组仅 5 个开发种子，置信区间较宽；同一开发集被用于调试和消融，不能当作独立最终测试。首版去掉条件输入后也成功一次，且成功种子不同，因此不能仅从首版成功率证明条件信息的有效性。保留的最终种子 1000–1019 未使用。", "",
          "小型原始日志副本位于 `runs/`，与原文件逐字节一致，来源与 SHA256 见 `RUNS_MANIFEST.json`。`demos/` 保留实际成功/失败视频，均为调试素材，不是最终汇报视频。", "",
          "## Jittor 与剩余证据", "", "Jittor 原生实现及 Linux 数值/训练入口已写好，目前未通过本机 Windows 编译。没有 Jittor 已通过对齐、训练曲线、GPU 性能或闭环成功率可报告。", "",
          "还需在同一 Linux 机器执行前向、索引、梯度、优化器一步和多步轨迹比较，再开展同小数据设置的训练与最终闭环对照。详见 `JITTOR_RUNTIME_STATUS.md`。", "",
          "当前全部实验均为单任务小数据设置，不等于论文的多任务 benchmark，也不证明自然语言或人形机器人泛化。", ""]
(reports / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"report": str(reports / "RESULTS.md"), "development_runs": list(evaluations)}, indent=2))
