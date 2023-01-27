#!/bin/bash

for seed in 43 44 45
do
    for algorithm in model_mnist model_mnist_model_robustness model_mnist_model_responsiveness model_mnist_image_robustness model_mnist_image_responsiveness
    do
        echo $seed $algorithm 
        python src/concept_vectors.py --algorithm $algorithm --class_name 0_number --target zebra --num_random_exp 10 --images_per_folder 100 --seed $seed
    done
done
