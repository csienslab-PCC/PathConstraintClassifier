# Train the three models with the in put dataset
# arg1 ($1) : input data file.
# arg2 ($2) : directrory of the model file.
# Example : ./run_train.sh ./split_bin-benchmark/train_benchmark.dat ./split_bin-benchmark/

./exp_utils.py -a train -i $1 -m 0:$2/DNN.model
./exp_utils.py -a train -i $1 -m 1:$2/DNN-alpha.model
./exp_utils.py -a train -i $1 -m 2:$2/DNN-beta.model
