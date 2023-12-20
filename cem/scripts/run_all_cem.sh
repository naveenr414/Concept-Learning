python experiments/extract_cem_concepts.py --num_epochs 50 --experiment_name cub --seed 42 --validation_epochs 1  --num_gpus 1
python experiments/extract_cem_concepts.py --num_epochs 50 --experiment_name mnist --seed 42 --validation_epochs 1  --num_gpus 1 --sample_train 0.05 --sample_test 0.05 --sample_valid 0.05
python experiments/extract_cem_concepts.py --num_epochs 50 --experiment_name dsprites --seed 42 --validation_epochs 1  --num_gpus 1
python experiments/extract_cem_concepts.py --num_epochs 50 --experiment_name chexpert --seed 42 --validation_epochs 1  --num_gpus 1