for seed in 43 44 45
do
    python extract_cem.py --experiment_name mnist --num_gpus 1 --num_epochs 50 --validation_epochs 25 --seed $seed
done