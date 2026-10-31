"""Bounded paired conditional-policy training and DEVELOPMENT rollouts only."""
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=4.95)
    args = parser.parse_args()
    if sys.platform != "linux" or args.epochs < 1 or not 0 < args.max_hours <= 5:
        parser.error("Linux; positive epochs; max-hours <= 5")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    tag = "policy_gpu_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = ROOT / "outputs" / tag
    output.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), NVIDIA_TF32_OVERRIDE="0", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="4")
    for key, folder in (("TMPDIR", "tmp"), ("PIP_CACHE_DIR", "pip"), ("MPLCONFIGDIR", "matplotlib")):
        path = ROOT / "cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    report = {"status": "running", "run": tag, "epochs": args.epochs,
        "started_utc": datetime.now(timezone.utc).isoformat(), "stages": [], "max_hours": args.max_hours,
        "scope": "single reach, frozen native-trained tokenizer, paired from-scratch policies, development seeds only",
        "policy_init": "validated full seed456 NPZ; same weights, not a pretrained policy",
        "precision": "FP32, TF32 off, ordinary exact-GELU Torch Encoder path", "server_shutdown": False}

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "workflow.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output / "workflow.json")

    def stage(name, command, cap=3600):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Policy workflow deadline reached")
        record = {"name": name, "command": [str(x) for x in command], "status": "running", "log": name + ".log",
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        report["stages"].append(record)
        save()
        print("START " + name, flush=True)
        tick = time.monotonic()
        with (output / record["log"]).open("wb") as log:
            child = subprocess.Popen(record["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=min(cap, remaining))
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                record["status"] = "interrupted_or_timed_out"
                save()
                raise
        record.update(status="passed" if code == 0 else "failed", exit_code=code, elapsed_seconds=time.monotonic()-tick)
        save()
        print(record["status"].upper() + " " + name, flush=True)
        if code:
            raise RuntimeError(f"{name} failed: {output / record['log']}")

    torch = ROOT / ".venv-torch-linux/bin/python"
    jt = ROOT / ".venv-jittor-linux/bin/python"
    native = [jt, "-u", "scripts/run_jittor_linux.py", "--device", "cuda"]
    token_folder = ROOT / "outputs/gpu_20261028_104533_tokenizer_jittor"
    token_weights, token_config = token_folder / "best_weights.npz", token_folder / "config.json"
    shared = output / "tokenizer_shared"
    token_pt = shared / "tokenizer_shared.pt"
    initial = ROOT / "outputs/fixtures/policy_full_seed456/weights.npz"
    common_token = ["--tokenizer-weights", token_weights, "--tokenizer-config", token_config]
    report.update(tokenizer_weights=str(token_weights), tokenizer_config=str(token_config))
    try:
        save()
        stage("shared_tokenizer", [torch, "scripts/prepare_policy_stage.py", "--framework", "torch", "--prepare-checkpoint",
              *common_token, "--output", shared], 120)
        stage("labels_torch", [torch, "scripts/prepare_policy_stage.py", "--framework", "torch", *common_token,
              "--tokenizer-checkpoint", token_pt, "--output", output / "labels_torch"], 300)
        stage("labels_jittor", [*native, "scripts/prepare_policy_stage.py", "--framework", "jittor", *common_token,
              "--output", output / "labels_jittor"], 600)
        for split in ("train", "validation"):
            stage("labels_compare_" + split, [torch, "scripts/compare_npz.py", output / "labels_torch" / (split+".npz"),
                  output / "labels_jittor" / (split+".npz"), "--output", output / ("labels_"+split+"_comparison.json")], 60)
        plan, smoke_plan = output / "batch_plan.npz", output / "smoke_batch_plan.npz"
        for name, epochs, destination in (("shared_plan", args.epochs, plan), ("smoke_plan", 1, smoke_plan)):
            stage(name, [torch, "scripts/create_batch_plan.py", "--epochs", epochs, "--batch-size", "16",
                  "--seed", "20261028", "--output", destination], 60)
        common = ["--device", "cuda", "--batch-size", "16", "--dropout", "0", "--seed", "20000"]
        native_common = [*common_token, "--initial-policy-weights", initial, *common]
        stage("native_training_smoke", [*native, "scripts/train_policy_jittor.py", *native_common,
              "--epochs", "1", "--batch-plan", smoke_plan, "--name", tag + "_native_smoke"])
        stage("train_torch", [torch, "-u", "scripts/train_policy.py", "--tokenizer", token_pt,
              "--initial-weights", initial, *common, "--epochs", args.epochs, "--batch-plan", plan,
              "--name", tag + "_torch"])
        stage("train_jittor", [*native, "scripts/train_policy_jittor.py", *native_common,
              "--epochs", args.epochs, "--batch-plan", plan, "--name", tag + "_jittor"])
        # Jittor env stays independent from Torch; simulation dependencies have no Torch requirement.
        stage("native_simulation_dependencies", [jt, "-m", "pip", "install", "-r", "requirements/simulation.txt", "-c", "requirements/jittor.txt"], 900)
        stage("native_environment_check", [jt, "-c", "import importlib.util,mujoco,metaworld,imageio_ffmpeg; assert importlib.util.find_spec('torch') is None; print('Independent native simulation environment: OK')"], 120)
        simulation_probe = "from quest_repro.simulation import ReachSimulation; import sys; s=ReachSimulation(9999); o,_=s.reset(); print({k:v.shape for k,v in o.items()}); s.close(); assert 'torch' not in sys.modules"
        stage("native_render_smoke", [jt, "-c", simulation_probe], 120)
        for framework, run_name in (("torch", tag+"_torch"), ("jittor", tag+"_jittor")):
            run = ROOT / "outputs" / run_name
            if framework == "torch":
                command = [torch, "scripts/evaluate_policy.py", "--policy", run/"best.pt", "--tokenizer", token_pt]
            else:
                command = [*native, "scripts/evaluate_policy_jittor.py", "--policy-weights", run/"best_weights.npz",
                           "--policy-config", run/"config.json", *common_token]
            stage("development_"+framework, [*command, "--device", "cuda", "--seed-set", "development",
                  "--episodes", "5", "--videos", "2", "--action-horizon", "8", "--max-steps", "150",
                  "--name", run_name+"_dev5"], 1200)
            stage("plot_"+framework, [jt, "scripts/plot_policy_run.py", run], 120)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
        print(report["error"], flush=True)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("REPORT " + str(output / "workflow.json"), flush=True)
        print("Server remains ON; final reserved seeds were not used.", flush=True)


if __name__ == "__main__":
    main()
