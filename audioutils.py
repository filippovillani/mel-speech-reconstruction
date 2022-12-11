import torch
import numpy as np
import librosa 

import config

def melspectrogram(audio: np.ndarray,
                   sr: int = 16000,
                   n_mels: int = 96, 
                   n_fft: int = 1024, 
                   hop_len: int = 256)->np.ndarray:
    
    melspectrogram = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=np.abs(audio),
                                                                            sr = sr, 
                                                                            n_fft = n_fft, hop_length = hop_len,
                                                                            n_mels = n_mels), 
                                             ref=np.max, 
                                             top_db=80.) / 80. + 1.
    
    
    return melspectrogram

def spectrogram(audio: np.ndarray,
                n_fft: int = 1024,
                hop_len: int = 256)->np.ndarray:
    
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y=audio, 
                                                              n_fft=n_fft,
                                                              hop_length=hop_len)), 
                                          ref=np.max, 
                                          top_db=80.) / 80. + 1.
    
    return spectrogram
