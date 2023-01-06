#!/bin/bash

for seed in 43 44 45
do

    python src/concept_vectors.py --algorithm tcav_mnist --class_name 0_color --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 1_color --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 2_color --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 3_color --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 4_color --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed

    python src/concept_vectors.py --algorithm tcav_mnist --class_name 0_number --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 1_number --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 2_number --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 3_number --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed
    python src/concept_vectors.py --algorithm tcav_mnist --class_name 4_number --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed

    python src/concept_vectors.py --algorithm tcav_mnist --class_name spurious --target zebra --num_random_exp 10 --images_per_folder 500 --seed $seed

done