# Train single model on three datasets.
# arg1 ($1) : model_id. (DNN: 0, DNN-alpha: 1, DNN-beta: 2)

./exp_utils.py -a train -i ./split_bin-angr/train_angr.dat -m $1:./split_bin-angr
./exp_utils.py -a train -i ./split_bin-klee/train_klee.dat -m $1:./split_bin-klee
./exp_utils.py -a train -i ./split_bin-benchmark/train_benchmark.dat -m $1:./split_bin-benchmark
