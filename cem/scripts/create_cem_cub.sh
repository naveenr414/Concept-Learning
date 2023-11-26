#!/bin/bash

echo "Creating CUB!"

for seed in 43 44 45
do
    for experiment_name in cub cub_image_robustness cub_image_responsiveness cub_model_robustness cub_model_responsiveness
    do
        echo $seed $experiment_name
        sbatch --partition=ampere --account=COMPUTERLAB-SL3-GPU --time=4:00:00 --gres=gpu:1 -N 1 -o "runs/${seed}_${experiment_name}.txt" -e "runs/error_${seed}_${experiment_name}.txt" scripts/run_cem.sh $experiment_name $seed
    done
done
