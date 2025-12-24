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
REPO_DIR="/project/6083487/yuvraj/perplexity-stock"          # <-- your repo on Nibi
VENV_DIR="$REPO_DIR/.venv"                                   # <-- venv path
DATA_DIR="$REPO_DIR/yf_data"                                 # <-- folder containing your yfinance CSVs
RUN_NAME="ppo_aapl_$(date +%Y%m%d_%H%M%S)"                    # auto run name
# ===============================

# Activate venv
source "$VENV_DIR/bin/activate"

cd "$REPO_DIR"

# Make sure output dirs exist
mkdir -p logs runs saves

# Helpful runtime env vars
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: make Torch a bit less noisy
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
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# ---- Run PPO ----
python -u train_ppo.py \
  -r "$RUN_NAME" \
  --data "$DATA_DIR" \
  --cuda \
  --reward_mode step_logret \
  --volumes \
  --bars 10 \
  --rollout_steps 2048 \
  --minibatch 256 \
  --epochs 10 \
  --lr 1e-3 \
  --val_every_rollouts 10 \
  --save_every_rollouts 10

echo "Job finished on: $(date)"
ls -lh saves | tail -n 50 || true
