#!/bin/bash

echo "Creating chexpert!"

for seed in 43 44 45
do
    for experiment_name in chexpert chexpert_image_robustness chexpert_image_responsiveness 
    do
        echo $seed $experiment_name
        sbatch --partition=ampere --account=COMPUTERLAB-SL2-GPU --time=30:00 --gres=gpu:1 -N 1 -o "runs/${seed}_${experiment_name}.txt" -e "runs/error_${seed}_${experiment_name}.txt" scripts/run_cem.sh $experiment_name $seed
    done
done
