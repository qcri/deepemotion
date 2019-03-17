#
# Author: yasser hifny
#

import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
import codecs
from keras.preprocessing import sequence
from keras.utils import np_utils 
import cPickle as pickle
from keras.layers import Embedding , Dense, Input,LSTM,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import  SpatialDropout1D,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import RMSprop, Adam, Nadam
from keras.layers import Conv1D, MaxPooling1D
from keras.engine.topology import Layer
from keras import initializers
from sklearn.model_selection import KFold
from keras.preprocessing import sequence
from keras.utils import np_utils 
import keras
from keras import optimizers
from keras.models import load_model
reload(sys)
sys.setdefaultencoding('utf-8')
import glob
import os
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
#import tensorflow as tf
from keras.callbacks import Callback
import warnings
#import tensorflow as tf
import random
random.seed(9001)

from sklearn.model_selection import KFold

LSTM_UNITS=64
DENSE_UNITS=64

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


# getting the number of GPUs 
# def get_available_gpus():
   # local_device_protos = device_lib.list_local_devices()
   # return [x.name for x in local_device_protos if x.device_type    == 'GPU']
# num_gpu = len(get_available_gpus())
# print "GPU count: ", num_gpu


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files,sample_weight,feat_norm_file,  batch_size=32, min_length=0, max_length=20000,
                 n_classes=7, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        _, self.list_files,self.list_label,self.sample_weight = self.sort_by_duration(list_files,min_length,sample_weight)
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_length = max_length
        self.feats_mean = np.load(feat_norm_file)['mean']
        self.feats_std = np.load(feat_norm_file)['std']
        self.on_epoch_end()
        print "generator is based on ", len(self.list_files),"files"
		

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find data for batch
        batch_data  = [self.normalize(np.load(self.list_files[k])['x']) for k in indexes]
        batch_label = [self.list_label[k] for k in indexes]
        W = [self.sample_weight[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch_data,batch_label)
		
        if sum(W)==len(W):		
            return X, y
        else:
            return X, y,np.asarray(W)
		

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)
		
    def __data_generation(self, batch_data,batch_label):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        max_length = min(self.max_length,max([ len(f) for f in batch_data]))
        #max_length = 1553
        X = sequence.pad_sequences( batch_data,
										maxlen=max_length,
										dtype='float32' )
        y = keras.utils.to_categorical(batch_label, num_classes=self.n_classes)

        return X, y
	
    def sort_by_duration(self,list_files, min_length, sample_weight):
        dialect_class_map = {'E00' : 1, 'E01' : 2, 'E02' : 3, 'E03' : 4, 'E04' : 5, 'E05':6}
        list_durations=[]
        list_labels=[]
        list_sample_weight=[]
        list_files_filtered=[]		
        for i,f in enumerate(list_files):
            length=len(np.load(f)['x'])
            w=sample_weight[i]
            if length >min_length:
                list_files_filtered.append(f)
                list_durations.append(length)
                list_labels.append(dialect_class_map[f.split('/')[11]] )
                list_sample_weight.append(w)
				
        return zip(*sorted(zip(list_durations, list_files_filtered, list_labels,list_sample_weight),key=lambda x: x[0]))	


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel', 
                                  shape=(input_shape[-1],),
                                  initializer='normal',
                                  trainable=True)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])		


seq_length = 6000
test_seq_length = 30000
batch_size=32
feat_dim=13
dropout_rate=0.2
print ("max  seq length", seq_length)
out_path='emotion_work_compact_cnn_big_adam_mfcc_essentia'


folder_path = '/data/sls/qcri-scratch/amali/exp/emotion_github/KSU_Emotions/data/SPEECH/Phase_*/E0*/*_mfcc_no_cvmn.npz'
all_feat_files=glob.glob(folder_path)


mean_var_file="mean_std_mfcc_no_cvmn.npz"

# create output dir
if not os.path.exists(out_path):
    os.makedirs(out_path)


kf = KFold(n_splits=5)
shuffle_indices = np.random.permutation(np.arange(len(all_feat_files)))
# Shuffle data
all_feat_files = np.array(all_feat_files)[shuffle_indices]
train_len = int(len(all_feat_files) * 0.8)
x_train = np.array(all_feat_files)[:train_len]
x_test = np.array(all_feat_files)[train_len:]
#print x_test

results=[]
fold=0
for train_index, test_index in kf.split(np.array(shuffle_indices)):
	print train_index
	print test_index
	x_train, x_test = np.array(all_feat_files)[train_index], np.array(all_feat_files)[test_index]
	
	
	# Generators
	train_list=x_train
	train_sample_weight= [1]*len(train_list) 

	training_generator = DataGenerator( train_list,train_sample_weight,mean_var_file,min_length=0, max_length=seq_length,
										batch_size=batch_size,shuffle=True)

	test_list=x_test									
	test_sample_weight= [1]*len(test_list) 
	validation_generator = DataGenerator(test_list, test_sample_weight,mean_var_file,min_length=0,max_length=test_seq_length,
										batch_size=1)

	test_generator = DataGenerator( test_list, test_sample_weight,mean_var_file,min_length=0,max_length=seq_length,
										batch_size=1)


	 







	input_layer = Input(shape=(None,feat_dim))
	conv=Dropout(dropout_rate)(input_layer)
	conv=Conv1D(500,
					 5,
					 padding='same',
					 activation='relu',
					 strides=1)(conv)
	conv=Dropout(dropout_rate)(conv)
	conv=Conv1D(500,
					 7,
					 padding='same',
					 activation='relu',
					 strides=2)(conv)
	conv=Dropout(dropout_rate)(conv)
	conv=Conv1D(500,
					 1,
					 padding='same',
					 activation='relu',
					 strides=2)(conv)
	conv=Dropout(dropout_rate)(conv)				 
	conv=Conv1D(500,
					 1,
					 padding='same',
					 activation='relu',
					 strides=1)(conv)
	conv=Dropout(dropout_rate)(conv)				 

	common=GlobalMaxPooling1D()(conv)
	dense = (Dense(500, activation='relu'))(common)
	dense=Dropout(dropout_rate)(dense)				 
	#dense = (Dense(600, activation='relu'))(dense)
	#dense=Dropout(dropout_rate)(dense)

	out = (Dense(7, activation='softmax'))(dense)

	#with tf.device('/cpu'):
	model = Model(input=input_layer, output=out)            
	#multi_model = multi_gpu_model(model, gpus=num_gpu)			


	# global_step = tf.Variable(0, trainable=False)
	# learning_rate = tf.train.exponential_decay(0.001, global_step,
											   # 50000, 0.98, staircase=True)
	# sgd = optimizers.TFOptimizer(tf.train.GradientDescentOptimizer(learning_rate))
#	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	#print(multi_model.summary())

	earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=40,verbose=1, mode='auto')
	checkpoint = MultiGPUCheckpointCallback(out_path+'/fold'+str(fold)+'-{epoch:03d}-{val_acc:.5f}.hdf5', model, save_best_only=True,monitor='val_acc',mode='max')
	model.fit_generator(generator=training_generator, validation_data=validation_generator,
			  epochs=1000,
			  verbose=1,
			  callbacks=[checkpoint,earlystop],
			  use_multiprocessing=True,
			  workers=6)
	## test
	output_model_file_list=glob.glob(out_path+'/fold'+str(fold)+'*.hdf5')		  
	list_index=[int(file.split('-')[1]) for file in output_model_file_list]
	index_list,file_list=zip(*sorted(zip(list_index, output_model_file_list),key=lambda x: x[0]))
	model_name=file_list[-1]
	#print index_list,file_list

	print model_name
	#model = load_model(model_name)
	model.load_weights(model_name)
	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
	score=model.evaluate_generator(test_generator,workers=1)
	print("Loss: ", score[0], "Accuracy: ", score[1])
	fold=fold+1
	results.append(score[1])

print('CV Test accuracy:', np.mean(np.array(results)))	
	
		  

