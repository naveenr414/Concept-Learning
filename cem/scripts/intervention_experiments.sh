#!/bin/bash

for algorithm in labels concept2vec normal random 
do
    sbatch --partition=cclake --account=COMPUTERLAB-SL3-CPU --mem=8g --time=3:00:00 -o "runs/intervention_${algorithm}.txt" -e "runs/error_intervention_${algorithm}.txt" scripts/intervention_helper.sh $algorithm 
done
