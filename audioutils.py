import numpy as np
import librosa 

def melspectrogram(audio: np.ndarray,
                   sr: int = 16000,
                   n_mels: int = 96, 
                   n_fft: int = 1024, 
                   hop_len: int = 256)->np.ndarray:
    
    melspectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio,
                                                                        sr = sr, 
                                                                        n_fft = n_fft, 
                                                                        hop_length = hop_len,
                                                                        n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    
    return melspectrogram

def stft(audio: np.ndarray,
         n_fft: int = 1024,
         hop_len: int = 256)->np.ndarray:
    
    spectrogram = librosa.stft(y=audio, 
                               n_fft=n_fft,
                               hop_length=hop_len)
    
    return spectrogram