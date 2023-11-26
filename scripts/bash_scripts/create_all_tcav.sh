#!/bin/bash 

dataset=mnist 
seed=43
suffix=none

sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --time=2:00:00 -o "runs/${dataset}_${seed}_${suffix}.txt" -e "runs/error_${dataset}_${seed}_${suffix}" scripts/bash_scripts/create_one_tcav.sh ${dataset} ${seed} ${suffix}