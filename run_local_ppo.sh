#!/bin/bash
set -euo pipefail

# ====== LOCAL PATHS ======
REPO_DIR="$HOME/Desktop/perplexity-stock"              # adjust if your repo is elsewhere
VENV_DIR="$REPO_DIR/.venv"                             # local virtual environment
DATA_DIR="$REPO_DIR/yf_data"                           # local yfinance CSV folder
RUN_NAME="ppo_$(date +%Y%m%d_%H%M%S)"                  # auto run name
# ==========================

cd "$REPO_DIR"

# Activate venv
source "$VENV_DIR/bin/activate"

# Make sure output dirs exist
mkdir -p logs runs saves

# Helpful runtime env vars
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TORCH_CPP_LOG_LEVEL=ERROR

echo "Job started on: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Run name: $RUN_NAME"
echo "Data dir: $DATA_DIR"
echo "Python: $(which python)"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# ---- PPO Training ----
python -u train_ppo.py \
  -r "$RUN_NAME" \
  --data "$DATA_DIR" \
  --cuda \
  --split \
  --train_ratio 0.80 \
  --min_train 200 \
  --min_val 200 \
  --reward_mode close_pnl \
  --bars 10 \
  --rollout_steps 2048 \
  --minibatch 256 \
  --epochs 10 \
  --lr 3e-4 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_eps 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --max_grad_norm 0.5 \
  --target_kl 0.02 \
  --time_limit 1000 \
  --val_every_rollouts 10 \
  --save_every_rollouts 50 \
  --early_stop \
  --min_rollouts 50 \
  --patience 20 \
  --max_rollouts 500

echo "Job finished on: $(date)"
echo "Recent checkpoints:"
ls -lh saves | tail -n 50 || true
echo "Recent runs:"
ls -lh runs  | tail -n 50 || true
