for algorithm in vae cem labels concept2vec concatenate model tcav_dr tcav
do 
    python src/metrics.py --algorithm $algorithm --dataset mnist
done 
