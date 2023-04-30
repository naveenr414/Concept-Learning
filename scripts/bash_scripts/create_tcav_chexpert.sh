#!/bin/bash

for seed in 43 44 45
do
    for suffix in image_robustness image_responsiveness
    do
        for attribute in "Enlarged Cardiom" "Cardiomegaly" "Lung Lesion" "Lung Opacity" "Edema" "Consolidation" "Pneumonia" "Atelectasis" "Pneumothroax" "Pleural Effusion" "Pleural Other" "Fracture" "Support Devices"
        do
            echo $seed $suffix $attribute
            python src/create_vectors.py --algorithm chexpert --class_name $attribute --target zebra --num_random_exp 3 --images_per_folder 100 --seed $seed --suffix $suffix --dataset mnist 
        done
    done
done
