#!/bin/bash

for seed in 43 44 45
do
    for algorithm in model_cub model_cub_model_robustness model_cub_model_responsiveness model_cub_image_robustness model_cub_image_responsiveness
    do
        echo $seed $algorithm 
        python src/concept_vectors.py --algorithm $algorithm --class_name 0_number --target zebra --num_random_exp 10 --images_per_folder 100 --seed $seed
    done
done
