#!/bin/bash

# List of models and metrics to evaluate
MODELS=("lorav2")
METRICS=("fid" "clipt" "clipi" "lpips")

# Loop through each model and metric combination
for model in "${MODELS[@]}"; do

    for metric in "${METRICS[@]}"; do
        echo "Running the generation of Model: $model | Metric: $metric"
        
        # Run the generation script for the current model and metric
        sbatch run_generation.sh "$model" "$metric"
    done
done

echo "All jobs have been submitted to the SLURM queue."