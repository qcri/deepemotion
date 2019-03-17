#
# Author: yasser hifny
#

import sys
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import preprocessing
import glob

from keras.preprocessing import sequence
from keras.utils import np_utils 
#np.seterr(all='warn')
import cPickle as pickle
import librosa



def normalize(feature, mean, std, eps=1e-14):
	return (feature - mean) / (std + eps)


def compute_mean_var_small_scale(folder_path):

	all_frames=[]

	for filepath in sorted(glob.glob(folder_path)):
		print "processing file: ", filepath
		all_frames.extend(np.load(filepath)['x'])

	feats = np.vstack(all_frames)
	feats_mean = np.mean(feats, axis=0)
	feats_std = np.std(feats, axis=0)
	
	return feats_mean,feats_std

def compute_mean_var_large_scale(folder_path):

	N_MFCC_COEFFS=40
	N=0
	mean_acc = np.zeros((1,N_MFCC_COEFFS), dtype='float32')
	cov_acc= np.zeros((N_MFCC_COEFFS,N_MFCC_COEFFS), dtype='float32')
	for filepath in sorted(glob.glob(folder_path)):
		print "processing file: ", filepath
		data_file=(np.load(filepath)['x'])
		mean_acc=mean_acc+np.sum(data_file, axis=0)
		cov_acc=cov_acc + (data_file.T).dot(data_file)
		N=N+data_file.shape[0]
		
	mean=mean_acc/float(N)	
	cov=cov_acc/float(N-1) -(((mean.T).dot(mean)) * float(N)/float(N-1))
	return mean.flatten(),  np.sqrt(cov.diagonal())

def compute_mean_var_large_scale_maximum_likelihood(folder_path):

	N_MFCC_COEFFS=13
	N=0
	mean_acc = np.zeros((1,N_MFCC_COEFFS), dtype='float32')
	cov_acc= np.zeros((N_MFCC_COEFFS,N_MFCC_COEFFS), dtype='float32')
	for filepath in sorted(glob.glob(folder_path)):
		print "processing file: ", filepath
		data_file=(np.load(filepath)['x'])
		mean_acc=mean_acc+np.sum(data_file, axis=0)
		cov_acc=cov_acc + (data_file.T).dot(data_file)
		N=N+data_file.shape[0]
		
	mean=mean_acc/float(N)	
	cov=cov_acc/float(N) -(mean.T).dot(mean) 
	return mean.flatten(),  np.sqrt(cov.diagonal())
	

	

	

train_folder_path = '/data/sls/qcri-scratch/amali/exp/emotion_github/KSU_Emotions/data/SPEECH/Phase_*/E0*/*_mfcc_no_cvmn.npz'


#mean,std=compute_mean_var_small_scale(train_folder_path)
mean,std=compute_mean_var_large_scale_maximum_likelihood(train_folder_path)
np.savez("mean_std_mfcc_no_cvmn.npz", mean=mean, std=std)






	
