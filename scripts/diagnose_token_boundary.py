"""Locate full-corpus GPU token disagreements and replay ORIGINAL batch shapes.

Report-only mode imports neither framework. A native replay imports exactly one
framework and records pre-quantization values; it never changes token rules.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from quest_repro.config import TokenizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--cpu-alignment", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--framework", choices=("inspect", "torch", "jittor"), default="inspect")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unfused-encoder", action="store_true",
                        help="Diagnostic: bypass Torch 2.2 fused encoder without changing activation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior boundary diagnostics")
    args.output.mkdir(parents=True)
    records, batches = [], []
    for split in ("train", "validation"):
        with np.load(args.alignment / "all_tokens_torch" / (split + ".npz")) as t, np.load(args.alignment / "all_tokens_jittor" / (split + ".npz")) as j, np.load(args.cpu_alignment / "all_tokens_torch" / (split + ".npz")) as c:
            bad = np.argwhere(t["indices"] != j["indices"])
            record = {"split": split, "windows": len(t["indices"]), "tokens": t["indices"].size,
                      "mismatch_count": len(bad),
                      "torch_gpu_vs_cpu_token_mismatches": int((t["indices"] != c["indices"]).sum()),
                      "jittor_gpu_vs_cpu_token_mismatches": int((j["indices"] != c["indices"]).sum()),
                      "mismatches": []}
            for row, position in bad:
                record["mismatches"].append({"row": int(row), "position": int(position),
                    "episode": int(t["episode_id"][row]), "start": int(t["start"][row]),
                    "torch_gpu": int(t["indices"][row, position]),
                    "jittor_gpu": int(j["indices"][row, position]),
                    "torch_cpu": int(c["indices"][row, position])})
            for start in sorted(set(int(row) // args.batch_size * args.batch_size for row, _ in bad)):
                end = min(start + args.batch_size, len(t["indices"]))
                selected = [x for x in record["mismatches"] if start <= x["row"] < end]
                batches.append((split, start, t["actions"][start:end].copy(), selected))
            records.append(record)
    report = {"framework": args.framework, "device": args.device, "batch_size": args.batch_size,
              "unfused_encoder": args.unfused_encoder,
              "splits": records, "scope": "diagnostic only; unchanged quantization, weights, data and tolerances",
              "replayed_batches": []}

    if args.framework != "inspect":
        info = json.loads((args.fixture / "metadata.json").read_text(encoding="utf-8"))
        config = TokenizerConfig(**info["config"])
        if args.framework == "torch":
            import torch
            from quest_repro.tokenizer_torch import SkillVAE
            from quest_repro.training import load_npz_state
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = SkillVAE(config).to(args.device).eval()
            if args.unfused_encoder:
                for layer in model.encoder.layers:
                    layer.activation_relu_or_gelu = 0
            load_npz_state(model, args.fixture / "weights.npz")
            def replay(actions):
                with torch.no_grad():
                    projected = model.action_proj(torch.from_numpy(actions).to(args.device))
                    convolution = model.conv_block(projected)
                    positioned = model.add_positional_emb(convolution)
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(positioned.shape[1],device=args.device)
                    encoded = model.encoder(positioned,mask=mask,is_causal=True)
                    latent = model.vq.project_in(encoded)
                    bounded = model.vq.bound(latent)
                    codes,indices=model.vq(encoded)
                    values = {"action_projection":projected,"convolution":convolution,"encoder":encoded,
                              "latent":latent,"bounded":bounded,"indices":indices}
                    return {key:value.cpu().numpy() for key,value in values.items()}
        else:
            import jittor as jt
            from quest_repro.tokenizer_jittor import SkillVAE, strict_load_npz
            jt.flags.use_cuda = int(args.device == "cuda")
            model = SkillVAE(config)
            strict_load_npz(model,args.fixture / "weights.npz")
            model.eval()
            def replay(actions):
                with jt.no_grad():
                    details=model.forward_details(jt.array(actions))
                    bounded=model.vq.bound(details["fsq_project_in"])
                    values={"action_projection":details["action_projection"],"convolution":details["convolution"],
                            "encoder":details["encoder"],"latent":details["fsq_project_in"],"bounded":bounded,
                            "indices":details["indices"]}
                    return {key:value.numpy() for key,value in values.items()}
        for split,start,actions,selected in batches:
            arrays=replay(actions)
            filename=f"{split}_batch_{start}.npz"
            np.savez_compressed(args.output/filename,**arrays)
            entry={"split":split,"batch_start":start,"batch_shape":list(actions.shape),"file":filename,"positions":[]}
            for item in selected:
                row,position=item["row"]-start,item["position"]
                value=arrays["bounded"][row,position]
                entry["positions"].append({**item,"replay_index":int(arrays["indices"][row,position]),
                    "latent":arrays["latent"][row,position].tolist(),"bounded":value.tolist(),
                    "distance_to_half_integer":np.abs(value-(np.floor(value)+0.5)).tolist()})
            report["replayed_batches"].append(entry)
    if args.framework == "jittor":
        assert "torch" not in sys.modules
    (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
