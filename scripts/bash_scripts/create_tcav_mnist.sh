#!/bin/bash

for seed in 43 44 45
do
    for algorithm in tcav_mnist_image_robustness tcav_mnist_image_responsiveness 
    do
        for attribute in 0_color 1_color 2_color 3_color 4_color 0_number 1_number 2_number 3_number 4_number spurious 
        do 
            echo $seed $algorithm $attribute
            python src/concept_vectors.py --algorithm $algorithm --class_name $attribute --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
        done
    done
done