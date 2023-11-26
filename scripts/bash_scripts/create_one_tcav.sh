#!/bin/bash

dataset=$1
seed=$2
suffix=$3

python src/create_vectors.py --dataset ${dataset} --seed ${seed} --suffix ${suffix} --algorithm tcav