#!/bin/bash
#SBATCH --job-name=ti_gen
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=/work/cvcs2026/stochastic_parrots/ktotaro/logs/job_%A_task_%a.out

MODEL=$1
METRIC=$2

mkdir -p /work/cvcs2026/stochastic_parrots/ktotaro/logs

# source /homes/tuousername/cvcs2026/venv/bin/activate
source /homes/ktotaro/cvcs2026/venv/bin/activate

python /work/cvcs2026/stochastic_parrots/ktotaro/evaluation/metrics/gen_parallel_ti.py --model "$MODEL" --metric "$METRIC"
