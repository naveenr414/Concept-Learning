#!/bin/bash

echo "Creating dSprites!"

for seed in 43 44 45
do
    for experiment_name in dsprites_image_responsiveness #dsprites dsprites_image_robustness dsprites_image_responsiveness 
    do
        echo $seed $experiment_name
        sbatch --partition=ampere --account=COMPUTERLAB-SL2-GPU --time=15:00 --gres=gpu:1 -N 1 -o "runs/${seed}_${experiment_name}.txt" -e "runs/error_${seed}_${experiment_name}.txt" scripts/run_cem.sh $experiment_name $seed
    done
done
