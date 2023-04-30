#!/bin/bash

for seed in 43 44 45
do
    for suffix in none image_robustness image_responsiveness
    do
        for attribute in is_white is_square is_ellipse is_heart is_scale_0.5 is_scale_0.6 is_scale_0.7 is_scale_0.8 is_scale_0.9 is_scale_1 is_orientation_0 is_orientation_90 is_orientation_180 is_orientation_270 is_x_0 is_x_16 is_y_0 is_y_16
        do
            echo $seed $suffix $attribute
            python src/create_vectors.py --algorithm tcav --class_name $attribute --target zebra --num_random_exp 3 --images_per_folder 100 --seed $seed --suffix $suffix --dataset mnist 
        done
    done
done
