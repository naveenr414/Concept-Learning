for algorithm in model 
do 
    python src/metrics.py --algorithm $algorithm --dataset mnist
done 
