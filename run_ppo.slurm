#!/bin/bash
#SBATCH --job-name=ppo-trade
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ====== CHANGE THESE PATHS ======
REPO_DIR="/project/6083487/yuvraj/perplexity-stock"     # your repo on Nibi
VENV_DIR="$REPO_DIR/.venv"                              # venv path
DATA_DIR="$REPO_DIR/yf_data"                            # folder containing yfinance CSVs
RUN_NAME="ppo_$(date +%Y%m%d_%H%M%S)"                   # auto run name
# ===============================

cd "$REPO_DIR"

# Activate venv
source "$VENV_DIR/bin/activate"

# Make sure output dirs exist
mkdir -p logs runs saves

# Helpful runtime env vars
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CPP_LOG_LEVEL=ERROR

echo "Job started on: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "GPUs: ${SLURM_GPUS:-1}"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
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

# ---- Full run settings (finite + early-stop) ----
# Guarantees job terminates even if early stopping never triggers:
#   - max_rollouts hard cap
#   - early_stop + patience for plateau
python -u train_ppo.py \
  -r "$RUN_NAME" \
  --data "$DATA_DIR" \
  --cuda \
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
