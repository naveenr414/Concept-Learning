for algorithm in labels tcav cem average concatenate model tcav_dr concept2vec
do 
    python src/metrics.py --algorithm $algorithm --dataset mnist
done 
