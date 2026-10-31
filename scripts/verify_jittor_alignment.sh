#!/usr/bin/env bash
# Execute on Linux with two isolated pre-existing project environments.
set -euo pipefail
quest_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$quest_root"
quest_device="${1:-cpu}"
case "$quest_device" in cpu|cuda) ;; *) echo 'device must be cpu or cuda' >&2; exit 2;; esac
quest_jt="$quest_root/.venv-jittor-linux/bin/python"
quest_torch="$quest_root/.venv-torch-linux/bin/python"
test -x "$quest_jt" && test -x "$quest_torch" || { echo 'Create isolated Linux environments first.' >&2; exit 1; }
quest_output="$quest_root/outputs/alignment_$(date +%Y%m%d_%H%M%S)_$quest_device"
mkdir "$quest_output"
"$quest_jt" scripts/run_jittor_linux.py --device "$quest_device" scripts/jittor_cpu_probe.py --device "$quest_device" --output "$quest_output/probe.json" 2>&1 | tee "$quest_output/probe.log"
test -s "$quest_output/probe.json" || { echo 'Jittor did not produce a successful probe. Keep logs; rerun after resolving it.' >&2; exit 1; }
for quest_case in tokenizer_tiny_seed123 tokenizer_full_trained policy_tiny_seed456 policy_full_seed456; do
    quest_fixture="$quest_root/outputs/fixtures/$quest_case"
    test -s "$quest_fixture/metadata.json" || { echo "Missing reference: $quest_fixture" >&2; exit 1; }
    quest_script="scripts/run_jittor_fixture.py"
    if [[ "$quest_case" == policy_* ]]; then quest_script="scripts/run_jittor_policy_fixture.py"; fi
    "$quest_jt" scripts/run_jittor_linux.py --device "$quest_device" "$quest_script" "$quest_fixture" --output "$quest_output/$quest_case" --device "$quest_device" 2>&1 | tee "$quest_output/$quest_case.log"
    for quest_field in features gradients; do
        "$quest_torch" scripts/compare_npz.py "$quest_fixture/$quest_field.npz" "$quest_output/$quest_case/$quest_field.npz" --output "$quest_output/${quest_case}_${quest_field}_comparison.json" --atol 0.00002 --rtol 0.0002
    done
    if [[ "$quest_case" == tokenizer_* ]]; then
        for quest_candidate in updated_weights optimizer_updated_weights; do
            "$quest_torch" scripts/compare_npz.py "$quest_fixture/updated_weights.npz" "$quest_output/$quest_case/$quest_candidate.npz" --output "$quest_output/${quest_case}_${quest_candidate}_comparison.json" --atol 0.00002 --rtol 0.0002
        done
    fi
done
quest_fixture="$quest_root/outputs/fixtures/tokenizer_tiny_seed123"
"$quest_torch" scripts/run_training_trace.py --framework torch --fixture "$quest_fixture" --output "$quest_output/trace_torch" --steps 20 --device "$quest_device" 2>&1 | tee "$quest_output/trace_torch.log"
"$quest_jt" scripts/run_jittor_linux.py --device "$quest_device" scripts/run_training_trace.py --framework jittor --fixture "$quest_fixture" --output "$quest_output/trace_jittor" --steps 20 --device "$quest_device" 2>&1 | tee "$quest_output/trace_jittor.log"
"$quest_torch" scripts/compare_npz.py "$quest_output/trace_torch/trace.npz" "$quest_output/trace_jittor/trace.npz" --output "$quest_output/trace_trace_comparison.json" --atol 0.00002 --rtol 0.0002
# Keep the strict failure visible; only a documented attention K-bias gauge
# equivalence may proceed, with unchanged tolerances on every other coordinate.
if "$quest_torch" scripts/compare_npz.py "$quest_output/trace_torch/final_weights.npz" "$quest_output/trace_jittor/final_weights.npz" --output "$quest_output/trace_final_weights_comparison.json" --atol 0.00002 --rtol 0.0002; then
    echo 'Training trace: strict all-parameter comparison passed.'
else
    "$quest_torch" scripts/compare_trace_parameters.py "$quest_output/trace_torch/final_weights.npz" "$quest_output/trace_jittor/final_weights.npz" --fixture "$quest_fixture" --output "$quest_output/trace_final_weights_equivalence.json"
    echo 'Training trace: K-bias gauge-equivalence passed; raw strict parameter failure is retained.'
fi
quest_fixture="$quest_root/outputs/fixtures/tokenizer_full_trained"
"$quest_torch" scripts/export_dataset_tokens.py --framework torch --fixture "$quest_fixture" --device "$quest_device" --output "$quest_output/all_tokens_torch" 2>&1 | tee "$quest_output/all_tokens_torch.log"
"$quest_jt" scripts/run_jittor_linux.py --device "$quest_device" scripts/export_dataset_tokens.py --framework jittor --fixture "$quest_fixture" --device "$quest_device" --output "$quest_output/all_tokens_jittor" 2>&1 | tee "$quest_output/all_tokens_jittor.log"
for quest_split in train validation; do
    "$quest_torch" scripts/compare_npz.py "$quest_output/all_tokens_torch/$quest_split.npz" "$quest_output/all_tokens_jittor/$quest_split.npz" --output "$quest_output/all_tokens_${quest_split}_comparison.json" --atol 0.00002 --rtol 0.0002
done
echo "Required comparisons passed; inspect any trace K-bias equivalence qualification. Reports: $quest_output"
echo 'This does not verify long training curves, full optimizer-state resume, CUDA performance or policy success.'
