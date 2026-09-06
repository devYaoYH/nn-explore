#!/bin/bash
# Bootstrap this repo's Python environment on a fresh GPU box (tested on
# Vast.ai instances). Detects GPU compute capability and picks a matching
# torch CUDA build automatically -- see the "Environment notes" section of
# README.md for why this matters (Blackwell GPUs silently produce garbage
# with an old CUDA build instead of erroring).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 .venv
source .venv/bin/activate

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 || echo "")"
echo "Detected GPU compute capability: ${CC:-unknown}"

# Blackwell (cc >= 10.0, e.g. RTX 50-series, B200) needs CUDA >= 12.8 wheels.
# Older CUDA 12.4 builds install cleanly but fail at runtime with
# "no kernel image is available for execution on the device".
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
if [ -n "${CC:-}" ] && awk -v cc="$CC" 'BEGIN{exit !(cc+0 >= 10.0)}'; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi
echo "Installing torch from: $TORCH_INDEX"
uv pip install torch --index-url "$TORCH_INDEX"

uv pip install -r requirements.txt

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
