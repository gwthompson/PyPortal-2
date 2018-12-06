from scipy.fftpack import fft, fftfreq
from scipy.signal import hanning
import numpy as np
from feature_extraction import FeatureChannelsIndepend, FeatureChannelsDepend
from fext import FeatureExtractor
import math
import helpers
#import librosa as lbr

def get_spectrum(sig, params):
    N=len(sig)
    T=1.0/params.Fs
    sig = sig - np.mean(sig)
    NFFT = int(math.pow(2,int(NextPowerOfTwo(N))))
    if params.doHanning:
        sig = sig*hanning(N)
    yf = fft(sig, NFFT)
    yf = 2.0/N*np.abs(yf[1:NFFT//2])
    freqs = fftfreq(NFFT, T)
    freqs = freqs[1:NFFT//2]
    idx = np.where((freqs > params.lowFreq) * (freqs < params.highFreq))

    #import matplotlib.pyplot as plt
    #plt.semilogy(freqs, yf, '-b')

    return yf[idx]

def NextPowerOfTwo(number):
    # Returns next power of two following 'number'
    return math.ceil(math.log(number,2))

def feat_assorti(sig,params):
    sig = sig - np.min(sig)
    sig = sig/np.max(sig)
    sig_=list()
    sig_.append(sig)
    feature = FeatureChannelsIndepend(sig_)
    feature_vector = feature.calculate('HFD')
    return feature_vector

def feat_assorti_2(sig,params):
    sig = sig - np.min(sig)
    sig = sig/np.max(sig)
    sig_=np.zeros((1,sig.shape[0]))
    sig_[0] = sig
    feature = FeatureExtractor()
    feature_vector = feature.extract(sig_, ['PFD', 'HFD', 'Hurst', 'Shannon', 'SVDEn', 'Fisher', 'Hjorth', 'DFA'])
    return feature_vector

def feat_for_cnn(sig,params):
    sig = sig - np.mean(sig)
    sig = sig/(np.std(sig))
    return sig

def feat_ts(sig, params):
    return

def convertToImg(sig, params):
    # Cropping and converting to img
    imgSize = params.imgSize
    map = helpers.sensors_map(params.sensorsFile, imgSize, imgSize, 2)
    arr = helpers.eeg2rgb(sig, params.Fs, map, imgSize, imgSize, method='cubic')
    return np.dstack((arr[0], arr[1], arr[2]))

# def feat_sound(sig,params):
#     sig = sig - np.mean(sig)
#     return lbr.feature.spectral_centroid(sig,params.Fs)
#     #return lbr.feature.mfcc(sig,params.Fs)

