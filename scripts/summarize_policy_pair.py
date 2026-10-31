"""Summarize actual paired policy training and small development evaluations."""
import argparse
import csv
import json
import math
import os
from pathlib import Path


def interval(successes, count):
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
    return [center-radius, center+radius]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous paired summaries")
    root = Path(__file__).resolve().parents[1]
    job = args.job.resolve()
    workflow = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
    if workflow["status"] != "completed":
        raise ValueError("Workflow must complete before summarizing")
    runs, records = {}, {}
    for framework in ("torch", "jittor"):
        folder = root / "outputs" / (workflow["run"] + "_" + framework)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        development = json.loads((folder.with_name(folder.name+"_dev5") / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "complete" and development["seed_set"] == "development"
        with (folder / "metrics.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        best = min(rows, key=lambda row:float(row["validation_ce"]))
        successes = sum(bool(row["success"]) for row in development["episodes"])
        count = len(development["episodes"])
        records[framework] = {"epochs": len(rows), "best_epoch_by_validation_ce": int(best["epoch"]),
            "best_validation_ce": float(best["validation_ce"]),
            "best_checkpoint_autoregressive_action_mae": float(best["autoregressive_action_mae"]),
            "final_train_ce": float(rows[-1]["train_ce"]), "final_validation_ce": float(rows[-1]["validation_ce"]),
            "training_elapsed_seconds": summary["elapsed_seconds"], "development_successes": successes,
            "development_episodes": count, "development_success_rate": successes/count,
            "wilson_95_interval": interval(successes,count), "development_episode_records": development["episodes"]}
        runs[framework] = {"config":config,"rows":rows}
    assert runs["torch"]["config"]["batch_plan_sha256"] == runs["jittor"]["config"]["batch_plan_sha256"]
    assert runs["torch"]["config"]["config"] == runs["jittor"]["config"]["config"]
    args.output.mkdir(parents=True)
    report = {"scope":"same native-trained frozen tokenizer, full policies from same initial NPZ, paired20epoch training, development only",
        "runs": records, "batch_plan_sha256":runs["torch"]["config"]["batch_plan_sha256"],
        "final_seeds_used":False,
        "limitations":["five development seeds; not a statistical framework ranking", "single task and 30 demonstrations",
                       "best checkpoint selected by validation CE, not development success", "training curves are not identical parameter trajectories",
                       "final reserved test, publication and presentation delivery remain pending"]}
    (args.output / "policy_pair_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR",str(root/"cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,4))
    for framework,entry in runs.items():
        rows=entry["rows"]; epochs=[int(row["epoch"]) for row in rows]
        axes[0].plot(epochs,[float(row["train_ce"]) for row in rows],label=framework)
        axes[1].plot(epochs,[float(row["validation_ce"]) for row in rows],label=framework)
        axes[2].plot(epochs,[float(row["autoregressive_action_mae"]) for row in rows],label=framework)
    for axis,title,ylabel in zip(axes,["Training CE","Validation CE","Autoregressive action error"],["Cross entropy","Cross entropy","Mean absolute error"]):
        axis.set(xlabel="Epoch",ylabel=ylabel,title=title); axis.grid(alpha=0.2); axis.legend()
    figure.suptitle("Paired policies: same frozen native tokenizer, initialization and batch plan")
    figure.tight_layout()
    figure.savefig(args.output / "paired_policy_curves.png",dpi=180)
    figure.savefig(args.output / "paired_policy_curves.pdf")
    plt.close(figure)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
