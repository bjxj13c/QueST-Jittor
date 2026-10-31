"""Bounded Linux workflow: environment -> parity gates -> paired GPU training.

Run in tmux. Does not manage cloud resources, install drivers, or change keys.
Every stage is logged; first failure prevents all dependent stages.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=3)
    parser.add_argument("--training-epochs", type=int, default=10)
    args = parser.parse_args()
    if sys.platform != "linux" or not 0 < args.max_hours <= 6 or args.training_epochs < 1:
        parser.error("Linux only; 0 < max-hours <= 6; positive epochs")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    tag = "gpu_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = ROOT / "outputs" / tag
    output.mkdir()
    report = {"status": "running", "run": tag, "started_utc": datetime.now(timezone.utc).isoformat(),
              "max_hours": args.max_hours, "epochs": args.training_epochs, "stages": [],
              "scope": "single reach task; native Jittor runtime/parity and small-data paired training; no final rollout",
              "server_shutdown": False}
    state_file = output / "workflow.json"
    environment = os.environ.copy()
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    report["precision"] = "float32; NVIDIA_TF32_OVERRIDE=0 for both frameworks"
    report["torch_encoder_path"] = "ordinary layer path with exact GELU callable; Torch 2.2 fused eval path bypassed"
    report["jittor_parallel_op_compiler"] = 0
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("OMP_NUM_THREADS", "4")
    for key, subfolder in (("TMPDIR", "tmp"), ("PIP_CACHE_DIR", "pip"), ("MPLCONFIGDIR", "matplotlib")):
        path = ROOT / "cache" / subfolder
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(state_file)

    def stage(name, command, cap=3600):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Workflow time budget exhausted")
        record = {"name": name, "command": [str(x) for x in command], "status": "running",
                  "started_utc": datetime.now(timezone.utc).isoformat(), "log": name + ".log"}
        report["stages"].append(record)
        save()
        print("START " + name, flush=True)
        tick = time.monotonic()
        with (output / record["log"]).open("wb") as log:
            process = subprocess.Popen(record["command"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=min(cap, remaining))
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["status"] = "interrupted_or_timed_out"
                save()
                raise
        record.update(exit_code=code, elapsed_seconds=time.monotonic() - tick,
                      status="passed" if code == 0 else "failed")
        save()
        print(record["status"].upper() + " " + name, flush=True)
        if code != 0:
            raise RuntimeError(f"Stage {name} failed; inspect {output / record['log']}")

    jt = ROOT / ".venv-jittor-linux/bin/python"
    torch = ROOT / ".venv-torch-linux/bin/python"
    base = Path("/root/autodl-tmp/conda-envs/quest-torch/bin/python")
    runner = [jt, "-u", "scripts/run_jittor_linux.py"]
    fixture = ROOT / "outputs/fixtures/tokenizer_full_trained"
    try:
        save()
        if not jt.is_file() or not base.is_file():
            raise FileNotFoundError("Expected migrated Python environments are missing")
        config_result = subprocess.run([jt, "-c", "import sys; print(sys.base_prefix)"],
                                       check=True, capture_output=True, text=True, env=environment)
        config = Path(config_result.stdout.strip()) / "bin/python3.10-config"
        if not config.is_file():
            raise FileNotFoundError(f"Python config helper missing: {config}")
        environment["python_config_path"] = str(config)
        report["python_config_path"] = str(config)
        if not torch.is_file():
            # Inherit the already installed CUDA Torch packages without changing the Conda env.
            stage("torch_isolated_environment", [base, "-m", "venv", "--system-site-packages", ROOT / ".venv-torch-linux"], 120)
        stage("preflight", [jt, "scripts/linux_preflight.py", "--output", output / "preflight.json"], 60)
        stage("torch_cuda", [torch, "-c", "import torch, torchvision, h5py, numpy; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, numpy.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones(2,device='cuda'); print((x+x).cpu())"], 120)
        for device in ("cpu", "cuda"):
            stage("probe_" + device, runner + ["--device", device, "scripts/jittor_cpu_probe.py", "--device", device, "--output", output / ("probe_" + device + ".json")], 2400)
        for device in ("cpu", "cuda"):
            stage("alignment_" + device, ["bash", "scripts/verify_jittor_alignment.sh", device], 3600)
        plan = output / "batch_plan.npz"
        epochs = str(args.training_epochs)
        stage("shared_plan", [torch, "scripts/create_batch_plan.py", "--output", plan, "--epochs", epochs, "--batch-size", "16", "--seed", "20261028"], 60)
        common = ["--device", "cuda", "--epochs", epochs, "--batch-size", "16", "--dropout", "0", "--seed", "10000", "--batch-plan", plan, "--initial-weights", fixture / "weights.npz"]
        # This is paired fine-tuning from a trained tokenizer, not training from scratch.
        report["training_scope"] = "paired tokenizer fine-tuning from the same trained NPZ; dropout disabled"
        stage("tokenizer_torch", [torch, "-u", "scripts/train_tokenizer.py", "--name", tag + "_tokenizer_torch"] + common)
        stage("tokenizer_jittor", runner + ["--device", "cuda", "scripts/train_tokenizer_jittor.py", "--name", tag + "_tokenizer_jittor"] + common)
        for framework, interpreter in (("torch", [torch]), ("jittor", runner + ["--device", "cuda"])):
            stage("benchmark_" + framework, interpreter + ["scripts/benchmark_tokenizer.py", "--framework", framework,
                  "--fixture", fixture, "--device", "cuda", "--batch-size", "16", "--warmup", "10", "--repeats", "50",
                  "--output", output / ("benchmark_" + framework + ".json")], 600)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("Report: " + str(state_file), flush=True)
        print("The server remains ON and billed by the platform.", flush=True)


if __name__ == "__main__":
    main()
