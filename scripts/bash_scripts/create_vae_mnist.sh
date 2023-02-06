#!/bin/bash

for seed in 43 44 45
do
    for dataset in mnist mnist_model_robustness mnist_model_responsiveness mnist_image_robustness mnist_image_responsiveness
    do
        echo $seed $dataset 
        python src/models.py --algorithm vae --seed $seed --dataset $dataset
    done
done
