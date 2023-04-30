#!/bin/bash

for seed in 43 44 45
do
    for suffix in image_robustness image_responsiveness
    do
        for attribute in 0_color 1_color 2_color 3_color 4_color 5_color 6_color 7_color 8_color 9_color 0_number 1_number 2_number 3_number 4_number 5_number 6_number 7_number 8_number 9_number spurious
        do
            echo $seed $suffix $attribute
            python src/create_vectors.py --algorithm tcav --class_name $attribute --target zebra --num_random_exp 3 --images_per_folder 100 --seed $seed --suffix $suffix --dataset mnist 
        done
    done
done
