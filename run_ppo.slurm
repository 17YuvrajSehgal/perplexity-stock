#!/bin/bash
#SBATCH --job-name=ppo-trade
#SBATCH --account=def-naser2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023 python/3.11 cuda/12.2

# ====== CHANGE THESE PATHS ======
REPO_DIR="/project/6083487/yuvraj/perplexity-stock"
VENV_DIR="$REPO_DIR/.venv"
DATA_DIR="$REPO_DIR/yf_data"
RUN_NAME="ppo_$(date +%Y%m%d_%H%M%S)"
# ===============================

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

mkdir -p logs runs saves

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

# --- Sanity check CUDA really works on the node ---
python - <<'PY'
import torch, sys
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA not available in this job. Check your PyTorch build / modules.")
    sys.exit(2)
print("device:", torch.cuda.get_device_name(0))
PY

# ---- FINAL "does it work?" run ----
# Goals:
# 1) Must run on GPU (we hard-fail above if not)
# 2) Must terminate (max_rollouts + total_steps + early_stop)
# 3) Must be comparable to your local run settings
# 4) Must generate a final evaluation print at the end using BEST checkpoint

python -u train_ppo.py \
  -r "$RUN_NAME" \
  --data "$DATA_DIR" \
  --cuda \
  --split \
  --train_ratio 0.80 \
  --min_train 300 \
  --min_val 300 \
  --bars 10 \
  --volumes \
  --reward_mode close_pnl \
  --time_limit 1000 \
  --rollout_steps 1024 \
  --minibatch 256 \
  --epochs 5 \
  --lr 3e-4 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_eps 0.2 \
  --value_coef 0.5 \
  --entropy_coef 0.01 \
  --max_grad_norm 0.5 \
  --target_kl 0.02 \
  --val_every_rollouts 10 \
  --save_every_rollouts 10 \
  --early_stop \
  --min_rollouts 50 \
  --patience 20 \
  --min_delta 0.01 \
  --max_rollouts 500 \
  --total_steps 10000000

echo
echo "=== FINAL EVAL (BEST checkpoint) ==="
python - <<PY
import os, torch
from finance_rl import data_yf, environ
from finance_rl.ppo_models import ActorCriticMLP
from finance_rl.ppo_validation import validation_run_ppo

run = "${RUN_NAME}"
ckpt = os.path.join("saves", f"ppo_{run}_best.pt")
assert os.path.exists(ckpt), f"Missing best checkpoint: {ckpt}"

prices = data_yf.load_many_from_dir("${DATA_DIR}")
train_p, val_p = data_yf.split_many_by_ratio(prices, train_ratio=0.80, min_train=300, min_val=300)

env = environ.StocksEnv(
    val_p,
    bars_count=10,
    volumes=True,
    extra_features=True,
    reset_on_close=False,
    reward_mode="close_pnl",
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCriticMLP(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n).to(dev)
model.load_state_dict(torch.load(ckpt, map_location=dev))
model.eval()

print("checkpoint:", ckpt)
print("device:", dev)
print(validation_run_ppo(env, model, episodes=200, device=str(dev), greedy=True))
PY

echo
echo "Job finished on: $(date)"
echo "Recent checkpoints:"
ls -lh saves | tail -n 50 || true
echo "Recent runs:"
ls -lh runs  | tail -n 50 || true
