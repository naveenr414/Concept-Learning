for algorithm in tcav_dr 
do 
    python src/metrics.py --algorithm $algorithm --dataset mnist
done 
