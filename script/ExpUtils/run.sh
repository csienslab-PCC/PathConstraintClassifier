# $1 : model_info, which specifies the model_id and model file path.
# Format: [model_id]:[model_file_path]
# Example: 0:./training/DNN.model

./exp_utils.py -a test_upload -i ./training2/ -m $1
./exp_utils.py -a test_upload -i ./training2/ -m $1 -logslct true
./exp_utils.py -a test_upload -i ./training2/ -m $1 -backup 1
./exp_utils.py -a test_upload -i ./training2/ -m $1 -backup 10
./exp_utils.py -a test_upload -i ./training2/ -m $1 -backup 50
./exp_utils.py -a test_upload -i ./training2/ -m $1 -backup 100
./exp_utils.py -a test_upload -i ./training2/ -m $1 -logslct true -backup 1
./exp_utils.py -a test_upload -i ./training2/ -m $1 -logslct true -backup 10
./exp_utils.py -a test_upload -i ./training2/ -m $1 -logslct true -backup 50
./exp_utils.py -a test_upload -i ./training2/ -m $1 -logslct true -backup 100
