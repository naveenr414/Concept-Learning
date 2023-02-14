#!/bin/bash

python src/models.py --algorithm vae --seed 44 --dataset cub
python src/models.py --algorithm vae --seed 45 --dataset cub
python src/models.py --algorithm vae_concept --seed 45 --dataset cub

python src/models.py --algorithm vae --seed 43 --dataset cub_model_robustness
python src/models.py --algorithm vae_concept --seed 43 --dataset cub_model_robustness
python src/models.py --algorithm vae_concept --seed 45 --dataset cub_model_robustness

python src/models.py --algorithm vae --seed 43 --dataset cub_model_responsiveness
python src/models.py --algorithm vae --seed 45 --dataset cub_model_responsiveness
python src/models.py --algorithm vae_concept --seed 43 --dataset cub_model_responsiveness



# for seed in 43 44 45
# do
#     for dataset in cub_image_robustness cub_image_responsiveness # cub cub_model_robustness cub_model_responsiveness 
#     do
#         for algorithm in vae vae_concept
#         do
#             echo $seed $dataset $algorithm
#             python src/models.py --algorithm $algorithm --seed $seed --dataset $dataset
#         done
#     done
# done
