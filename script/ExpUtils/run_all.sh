# Run the training & testing procedure ob the three dataset.

./run_train.sh ./split_bin-klee/train_klee.dat ./split_bin-klee/
./run_test.sh ./split_bin-klee/test_klee.dat ./split_bin-klee/
./run_train.sh ./split_bin-angr/test_angr.dat ./split_bin-angr/
./run_test.sh ./split_bin-angr/train_angr.dat ./split_bin-angr/
./run_train.sh ./split_bin-benchmark/train_benchmark.dat ./split_bin-benchmark/
./run_test.sh ./split_bin-benchmark/test_benchmark.dat ./split_bin-benchmark/
