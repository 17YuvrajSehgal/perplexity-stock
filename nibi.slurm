#!/bin/bash
#SBATCH --job-name=dqn-aapl-gpu
#SBATCH --account=def-naser2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# If Nibi requires a GPU partition on your setup, uncomment ONE of these:
# #SBATCH --partition=gpu
# #SBATCH --constraint=a100        # (example) adjust/remove based on available GPUs

module --force purge
module load StdEnv/2023 python/3.11
# Load CUDA if your PyTorch on Nibi is CUDA-enabled via modules.
# If your venv already has a CUDA-enabled torch wheel, you may not need this.
# Common options (pick the one that exists on Nibi):
# module load cuda/12.2
# module load cudnn

# Activate venv (update path)
source /project/6083487/yuvraj/perplexity-stock/.venv/bin/activate

# Go to repo (update path)
cd /project/6083487/yuvraj/perplexity-stock

mkdir -p logs runs saves

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Mem : $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS"
nvidia-smi || true

RUN_NAME="nibi_gpu_${SLURM_JOB_ID}"

# IMPORTANT: --cuda enables GPU usage in your train_model.py
python -u train_model.py \
  -r "$RUN_NAME" \
  --data "yf_data" \
  --cuda

echo "Job finished: $(date)"
ls -lh runs || true
