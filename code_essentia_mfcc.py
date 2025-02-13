#### this reproduces the way htk extracts MFCC with the default configuration:
# SOURCEFORMAT = WAV
# TARGETKIND = MFCC_0
# TARGETRATE = 100000.0
# SAVECOMPRESSED = T
# SAVEWITHCRC = T
# WINDOWSIZE = 250000.0
# USEHAMMING = T
# PREEMCOEF = 0
# NUMCHANS = 26
# CEPLIFTER = 22
# NUMCEPS = 12
# ENORMALISE = F
# HIFREQ=8000
import sys
#sys.path.append("/usr/local/lib/python2.7/site-packages")

import essentia
import essentia.standard as ess

import os
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import preprocessing
import glob

from keras.preprocessing import sequence
from keras.utils import np_utils 
np.seterr(all='warn')


folder_path = '/data/sls/qcri-scratch/amali/exp/emotion_github/KSU_Emotions/data/SPEECH/Phase_*/E0*/*.wav'
emotion_class_map = {'E00' : 1, 'E01' : 2, 'E02' : 3, 'E03' : 4, 'E04' : 5, 'E05':6}
emotion_data = []
emotion_label = []



def extractor(filename):

    fs = 44100
    audio = ess.MonoLoader(filename = filename, 
                                          sampleRate = fs)()
    # dynamic range expansion as done in HTK implementation
    audio = audio*2**15

    frameSize = 1102 # corresponds to htk default WINDOWSIZE = 250000.0 
    hopSize = 441 # corresponds to htk default TARGETRATE = 100000.0
    fftSize = 2048
    spectrumSize= fftSize//2+1
    zeroPadding = fftSize - frameSize

    w = ess.Windowing(type = 'hamming', #  corresponds to htk default  USEHAMMING = T
                        size = frameSize, 
                        zeroPadding = zeroPadding,
                        normalized = False,
                        zeroPhase = False)

    spectrum = ess.Spectrum(size = fftSize)

    mfcc_htk = ess.MFCC(inputSize = spectrumSize,
                        type = 'magnitude', # htk uses mel filterbank magniude
                        warpingFormula = 'htkMel', # htk's mel warping formula
                        weighting = 'linear', # computation of filter weights done in Hz domain
                        highFrequencyBound = 8000, # corresponds to htk default
                        lowFrequencyBound = 0, # corresponds to htk default
                        numberBands = 26, # corresponds to htk default  NUMCHANS = 26
                        numberCoefficients = 13,
                        normalize = 'unit_max', # htk filter normaliation to have constant height = 1  
                        dctType = 3, # htk uses DCT type III
                        logType = 'log',
                        liftering = 22) # corresponds to htk default CEPLIFTER = 22


    mfccs = []
    # startFromZero = True, validFrameThresholdRatio = 1 : the way htk computes windows
    for frame in ess.FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize , startFromZero = True, validFrameThresholdRatio = 1):
        spect = spectrum(w(frame))
        mel_bands, mfcc_coeffs = mfcc_htk(spect)
        #frame_energy = energy_func(frame)
        #mfccs.append(numpy.append(mfcc_coeffs, frame_energy))
        mfccs.append(mfcc_coeffs)
		
		
    return mfccs	

def normalize_vec(feature, feats_mean,feats_std, eps=1e-14):
        return (feature - feats_mean) / (feats_std + eps)	
	
def normalize( emotion_data):
	
	# compute mean and std
	all_frames=[]
	for  utt in emotion_data:
		all_frames.extend(utt)

	feats = np.vstack(all_frames)
	feats_mean = np.mean(feats, axis=0)
	feats_std = np.std(feats, axis=0)
	del all_frames
	
	# scale the data
	output=[]
	for  utt in emotion_data:
		scaled_utt=[normalize_vec(frame, feats_mean,feats_std) for frame in utt]
		output.append(scaled_utt)
		
	return output
	
	


for filepath in glob.glob(folder_path):	
	print ("processing ", filepath)
	frames=extractor(filepath)
	if len(np.argwhere(np.isnan(np.array(frames))))>0:
		print ("error nan in file:", filepath)
		continue
	feat_type="mfcc"	
	print ("save ", os.path.splitext(filepath)[0]+"_"+feat_type+"_no_cvmn")
	np.savez(os.path.splitext(filepath)[0]+"_"+feat_type+"_no_cvmn", x=np.array(frames))	

