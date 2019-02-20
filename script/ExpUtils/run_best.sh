# Test the performance of "Best" model with the in put dataset
# arg1 ($1) : input data file.
# Example : ./run_best.sh ./split_bin-benchmark/train_benchmark.dat

./exp_utils.py -a test_upload -i $1 -m 5:./DummyFile
