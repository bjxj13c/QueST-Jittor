#!/usr/bin/env bash
# Explicitly invoked setup only; does not rent/start resources or install drivers.
set -euo pipefail
quest_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
quest_framework="${1:-torch}"
quest_python="${2:-python3.10}"
quest_build="${3:-cpu}"
case "$quest_framework" in torch|jittor) ;; *) echo 'framework must be torch or jittor' >&2; exit 2;; esac
case "$quest_build" in cpu|cu121) ;; *) echo 'build must be cpu or cu121' >&2; exit 2;; esac
"$quest_python" -c 'import sys; assert sys.version_info[:2] == (3,10), "Provide an existing Python 3.10 executable"'
export PIP_CACHE_DIR="$quest_root/cache/pip"
export TMPDIR="$quest_root/cache/tmp"
export MPLCONFIGDIR="$quest_root/cache/matplotlib"
export HF_HOME="$quest_root/cache/huggingface"
export TORCH_HOME="$quest_root/cache/torch"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME"
quest_venv="$quest_root/.venv-$quest_framework-linux"
if [[ ! -x "$quest_venv/bin/python" ]]; then "$quest_python" -m venv "$quest_venv"; fi
quest_exec="$quest_venv/bin/python"
"$quest_exec" -m pip install pip==25.3 setuptools==75.8.2 wheel==0.45.1
if [[ "$quest_framework" == torch ]]; then
    "$quest_exec" -m pip install "torch==2.2.0+$quest_build" "torchvision==0.17.0+$quest_build" --index-url "https://download.pytorch.org/whl/$quest_build"
    "$quest_exec" -m pip install -r "$quest_root/requirements/cpu.txt"
else
    command -v g++ >/dev/null || { echo 'An existing g++ compiler is required. No system packages were installed.' >&2; exit 1; }
    "$quest_exec" -m pip install -r "$quest_root/requirements/jittor.txt"
fi
"$quest_exec" -m pip install --no-deps --no-build-isolation -e "$quest_root"
echo "Ready: $quest_exec (framework installation only; run preflight and tests next)"
