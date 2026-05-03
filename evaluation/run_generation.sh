#!/bin/bash
#SBATCH --job-name=sdxl_5k
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
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