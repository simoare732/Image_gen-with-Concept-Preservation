#!/bin/bash

MODELS=("tiv1")
METRICS=("fid" "clipt" "clipi" "lpips")

for model in "${MODELS[@]}"; do
    for metric in "${METRICS[@]}"; do
        echo "Submitting: model=$model | metric=$metric"
        sbatch run_generation_ti.sh "$model" "$metric"
    done
done

echo "All TI generation jobs submitted."
