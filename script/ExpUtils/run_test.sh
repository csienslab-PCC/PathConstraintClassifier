# Test the performance of three models with different option and upload the 
# results to a google spreadsheet.
#
# arg1 ($1) : input data file.
# arg2 ($2) : directrory of the model file.
# Example : ./run_test.sh ./split_bin-benchmark/train_benchmark.dat ./split_bin-benchmark/

./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model 
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -logslct true
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -backup 1
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -backup 10
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -backup 50
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -backup 100
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -logslct true -backup 1
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -logslct true -backup 10
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -logslct true -backup 50
./exp_utils.py -a test_upload -i $1 -m 0:$2/DNN.model -logslct true -backup 100
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model 
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -logslct true
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -backup 1
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -backup 10
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -backup 50
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -backup 100
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -logslct true -backup 1
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -logslct true -backup 10
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -logslct true -backup 50
./exp_utils.py -a test_upload -i $1 -m 1:$2/DNN-alpha.model -logslct true -backup 100
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -logslct true
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -backup 1
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -backup 10
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -backup 50
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -backup 100
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -logslct true -backup 1
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -logslct true -backup 10
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -logslct true -backup 50
./exp_utils.py -a test_upload -i $1 -m 2:$2/DNN-beta.model -logslct true -backup 100
