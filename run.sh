export CUDA_VISIBLE_DEVICES="1"

python code_essentia_mfcc.py
python compute_mean_var_mfcc_no_cvmn_essentia.py

KERAS_BACKEND=tensorflow  nohup python train_cnn_lstm_att_mfcc_essentia.py  >& ./log_train_cnn_lstm_att_mfcc_essentia.txt &
KERAS_BACKEND=tensorflow  nohup python train_cnn_mfcc_essentia.py  >& ./log_train_cnn_essentia.txt &
