python experiments/extract_cem_concepts.py --num_epochs 100 --experiment_name cub --seed 42 --validation_epochs 1  --num_gpus 1
python experiments/extract_cem_concepts.py --num_epochs 25 --experiment_name mnist --seed 42 --validation_epochs 1  --num_gpus 1 --sample_train 0.1 --sample_test 0.1 --sample_valid 0.1 --concept_loss_weight 1 --lr 0.0001
python experiments/extract_cem_concepts.py --num_epochs 10 --experiment_name dsprites --seed 42 --validation_epochs 1  --num_gpus 1 --lr 0.001
python experiments/extract_cem_concepts.py --num_epochs 100 --experiment_name chexpert --seed 42 --validation_epochs 1  --num_gpus 1


python experiments/extract_cem_concepts.py --correlation_rate 1.0 --num_epochs 5 --experiment_name mnistcorrelated --seed 42 --validation_epochs 1  --num_gpus 1 --sample_train 0.1 --sample_test 0.1 --sample_valid 0.1 --concept_loss_weight 1 --lr 0.0001 
