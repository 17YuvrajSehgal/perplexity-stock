#!/bin/bash
#SBATCH --job-name=rl_trading
#SBATCH --account=def-naser2
#SBATCH --output=slurm_outputs/rl_trading_%j.out
#SBATCH --error=slurm_outputs/rl_trading_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules (adjust based on your cluster's available modules)
# Uncomment and modify as needed for your cluster
module load python/3.13.2
module load cuda/12.
module load cudnn/9.13.1.26

# Activate virtual environment if using one
# If you have a conda environment:
# source activate trading_env
# Or if using venv:
source .venv/bin/activate

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Print Python and package versions
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Create output directories
mkdir -p slurm_outputs
mkdir -p outputs
mkdir -p data  # For storing data files if needed

# Note: If cluster has no internet access, ensure data files are available:
# - Place {ticker}_processed.csv in current directory, or
# - Place in data/ subdirectory, or
# - Use --data_file argument to specify path

# Default parameters (modify as needed)
TICKER=${TICKER:-AAPL}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-auto}

# Run the training script
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Parameters:"
echo "  Ticker: $TICKER"
echo "  Total Timesteps: $TOTAL_TIMESTEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "=========================================="

python main.py \
    --ticker $TICKER \
    --total_timesteps $TOTAL_TIMESTEPS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --output_dir outputs

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

