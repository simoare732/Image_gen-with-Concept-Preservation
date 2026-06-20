#!/bin/bash
#SBATCH --job-name=sdxl_5k
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

# Run 50 parallel tasks, each generating 100 images
#SBATCH --array=0-49

# Save logs in the logs directory, with a unique name for each task
#SBATCH --output=logs/job_%A_task_%a.out

# Take input
MODEL=$1
METRIC=$2

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the virtual environment
source /homes/saresta/cvcs2026/venv/bin/activate

# Run the generation script
python metrics/gen_parallel.py --model "$MODEL" --metric "$METRIC"