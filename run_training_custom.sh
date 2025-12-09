#!/bin/bash
# Custom SLURM submission script with parameter options
# Usage: sbatch run_training_custom.sh [ticker] [timesteps] [learning_rate] [batch_size]

# Set default values or use command line arguments
TICKER=${1:-AAPL}
TOTAL_TIMESTEPS=${2:-100000}
LEARNING_RATE=${3:-3e-4}
BATCH_SIZE=${4:-64}

# Create a unique job name
JOB_NAME="rl_trading_${TICKER}_${TOTAL_TIMESTEPS}"

# Submit the job with custom parameters
sbatch --job-name=$JOB_NAME \
       --export=TICKER=$TICKER,TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS,LEARNING_RATE=$LEARNING_RATE,BATCH_SIZE=$BATCH_SIZE \
       run_training.slurm

echo "Submitted job: $JOB_NAME"
echo "  Ticker: $TICKER"
echo "  Timesteps: $TOTAL_TIMESTEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"

