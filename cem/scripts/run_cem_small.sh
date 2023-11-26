#!/bin/bash

experiment_name=$1
seed=$2

num_gpus=0

LD_LIBRARY_PATH=../../anaconda3/lib python experiments/extract_cem_concepts.py --experiment_name $experiment_name --num_gpus $num_gpus --num_epochs 1 --validation_epochs 1 --seed $seed --concept_pair_loss_weight 0 --sample_train 0.05 --sample_test 0.05
