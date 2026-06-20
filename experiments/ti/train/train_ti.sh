#!/bin/bash
#SBATCH --job-name=ti_training
#SBATCH --account=cvcs2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=/work/cvcs2026/stochastic_parrots/logs/train_ti_%j.out
#SBATCH --error=/work/cvcs2026/stochastic_parrots/logs/train_ti_%j.err

mkdir -p /work/cvcs2026/stochastic_parrots/logs

cd /work/cvcs2026/stochastic_parrots
source /homes/$USER/cvcs2026/venv/bin/activate
python experiments/ti/train/launcher.py
