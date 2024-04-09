from scipy import signal
import librosa
import numpy as np
from os import path
from IPython import embed
import pickle
from python_speech_features.base import delta


class FeatureBase(object):

    def __init__(self, n_mel_bands, n_fft, sampling_rate, frame_length_ms, frame_shift_ms):
        self.n_mel_bands = n_mel_bands
        self.n_fft = n_fft
        self.sampling_rate = sampling_rate
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms

    def _stft_parameters(self):
        n_fft = self.n_fft
        hop_length = int(self.frame_shift_ms / 1000 * self.sampling_rate)
        win_length = int(self.frame_length_ms / 1000 * self.sampling_rate)
        return n_fft, hop_length, win_length

    def _stft(self, y, window, eps):
        n_fft, hop_length, win_length = self._stft_parameters()
        return np.abs(librosa.stft(y + eps, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True))

    def load_wav(self, path_dir):
        return librosa.core.load(path_dir, sr=self.sampling_rate)[0]

    @staticmethod
    def save_spectrogram_train(path_dir, data ,spectrogram, label, mean_vector, std_vector):
        with open(path.join(path_dir, data.split('/')[-1].replace('.wav', '.p')), 'wb') as f_name:
            pickle.dump({'feat': spectrogram, 'label' : label,'stat':{'mean': mean_vector, 'std':std_vector}}, f_name)
    @staticmethod
    def save_spectrogram_val_test(path_dir, data ,spectrogram, label):
        with open(path.join(path_dir, data.split('/')[-1].replace('.wav', '.p')), 'wb') as f_name:
            pickle.dump({'feat': spectrogram, 'label': label}, f_name)

class Mel(FeatureBase):

    def __init__(self, n_mel_bands, n_fft, sampling_rate, frame_length_ms,
                 frame_shift_ms, window , eps, fmin, fmax, htk):
        super(Mel, self).__init__(
            n_mel_bands=n_mel_bands,
            n_fft= n_fft,
            sampling_rate=sampling_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms)
        
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk
        self.window = window
        self.eps = eps

    def melspectrogram(self, y):
        
        spectrogram = self._stft(y, self.window, self.eps)
        mel_basis = librosa.filters.mel(sr=self.sampling_rate, n_fft=self.n_fft,
                                        n_mels=self.n_mel_bands, fmin=self.fmin, 
                                        fmax=self.fmax, htk=self.htk)

        feature_matrix = np.dot(mel_basis, spectrogram)
        feature_matrix = np.log(feature_matrix + self.eps)
        return feature_matrix

    def spec(self, y):
        spectrogram = np.log(self._stft(y, self.window, self.eps) + self.eps)
        return spectrogram


    def librosa_mel(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        feature_matrix = librosa.feature.melspectrogram(y + self.eps, sr=self.sampling_rate, n_fft=n_fft, 
                                         hop_length=hop_length, win_length=win_length, 
                                         n_mels=self.n_mel_bands, window='hamming') 

        return feature_matrix
        



class Mfcc(FeatureBase):

    def __init__(self, n_mel_bands, n_fft, sampling_rate, frame_length_ms, frame_shift_ms,
                 n_coeff, eps):
        super(Mfcc, self).__init__(
            n_mel_bands=n_mel_bands,
            n_fft= n_fft,
            sampling_rate=sampling_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms)

        self.n_coeff = n_coeff
        self.eps = eps

        
    def Mfcc_(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        mfcc_coef = librosa.feature.mfcc(y=y+self.eps, sr=self.sampling_rate, n_fft=n_fft, 
                                         n_mfcc=self.n_coeff, hop_length=hop_length, win_length=win_length, 
                                         n_mels=self.n_mel_bands)                               
        return mfcc_coef

    def DeltaMfcc(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        mfcc_coef = librosa.feature.mfcc(y=y+self.eps, sr=self.sampling_rate,n_fft=n_fft, 
                                         n_mfcc=self.n_coeff, hop_length=hop_length, win_length=win_length, 
                                         n_mels=self.n_mel_bands)

        delta_coef = delta(mfcc_coef.T, 3)
        delta2_coef = delta(delta_coef, 3)
        delta_mfcc_coef = np.vstack((mfcc_coef,delta_coef.T,delta2_coef.T))
        return delta_mfcc_coef
    