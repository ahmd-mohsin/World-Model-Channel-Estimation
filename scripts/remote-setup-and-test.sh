#!/usr/bin/env bash
#
# remote-setup-and-test.sh — RUN THIS ON THE GREENLAND INSTANCE (not your laptop).
# --------------------------------------------------------------------------------
# Creates the `wireless` venv on the GPU box, installs deps, and runs the
# encoder + target-encoder GPU tests/demos to confirm everything works on A100.
#
# Typical flow from your laptop:
#   ./scripts/greenland-auth.sh
#   ./scripts/greenland-connect.sh tunnel      # terminal 1 (keep open)
#   ./scripts/greenland-sync.sh up             # terminal 2
#   ./scripts/greenland-connect.sh ssh         # terminal 2 -> now on the box
#   cd ~/World-Model-Channel-Estimation && bash scripts/remote-setup-and-test.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==== GPU visibility ===="
nvidia-smi || { echo "nvidia-smi failed — no GPU?"; exit 1; }

echo "==== Python env ===="
PY=python3
if [ ! -d wireless ]; then
  $PY -m venv wireless
fi
source wireless/bin/activate
python -m pip install --upgrade pip --quiet

# The Greenland pytorch-base image already has CUDA torch; only add what's missing.
python - <<'EOF'
import importlib, subprocess, sys
need = []
for pkg, imp in [("numpy","numpy"),("transformers","transformers"),
                 ("huggingface_hub","huggingface_hub"),("matplotlib","matplotlib"),
                 ("scikit-learn","sklearn"),("pytest","pytest")]:
    try: importlib.import_module(imp)
    except ImportError: need.append(pkg)
try:
    import torch  # noqa
except ImportError:
    need.append("torch")
if need:
    print("installing:", need)
    subprocess.check_call([sys.executable,"-m","pip","install","--quiet",*need])
else:
    print("all deps present")
EOF

echo "==== Torch CUDA check ===="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
EOF

echo "==== Run test suite (CPU-correctness) ===="
python -m pytest implementation/ -q

echo "==== GPU smoke: encoder + target on CUDA ===="
python - <<'EOF'
import torch
from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.target_encoder import TargetEncoder

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = SSWMConfig(n_subcarriers=32, n_antennas=32, seq_len=8, horizon_k=4,
                 embed_dim=256, backbone="lwm", use_pretrained=True)
ctx = ContextEncoder(cfg).to(dev)
tgt = TargetEncoder(ctx, cfg).to(dev)
o = torch.randn(16, cfg.seq_len, 2, 32, 32, device=dev)
x = ctx(o); z = tgt(o)
print(f"[{dev}] o {tuple(o.shape)} -> x {tuple(x.shape)} | z~ {tuple(z.shape)} | x.device={x.device}")
assert x.shape == (16, cfg.seq_len, cfg.embed_dim)
print("GPU smoke OK")
EOF

echo "==== Done. Pull figures back with: ./scripts/greenland-sync.sh down ===="
