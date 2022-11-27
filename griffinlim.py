import numpy as np
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt

import config

def griffin_lim_base(spectrogram: np.ndarray, 
                     out_path: str, 
                     num_iter: int = 512, 
                     sr: int = 16000):
    
    out_audio_path = config.RESULTS_DIR / (str(out_path) + '.wav')
    X_init_abs = np.abs(spectrogram)
    X_init_phase = np.zeros(X_init_abs.shape)
    # X_init_phase = np.random.uniform(-np.pi, np.pi, size=X_init_abs.shape)
    X = X_init_abs * np.exp(1j * X_init_phase)
    
    for n in range(num_iter):
        _, X_hat = scipy.signal.istft(X)    # G+ cn
        _, _, X_hat = scipy.signal.stft(X_hat) # G G+ cn
        X_phase = np.angle(X_hat)
        X = X_init_abs * np.exp(1j * X_phase)   # Pc1(Pc2(cn-1))
    
    _, x = scipy.signal.istft(X)
    sf.write(out_audio_path, x, samplerate=sr)
    
    return x
    
    
# def fast_griffin_lim(spectrogram: np.ndarray, 
#                     out_path: str, 
#                     num_iter: int = 512,
#                     alpha: float = 0.99, 
#                     sr: int = 16000):

#     out_audio_path = config.RESULTS_DIR / (str(out_path) + '.wav')
#     X_init_abs = np.abs(spectrogram)
#     X_init_phase = np.random.uniform(-np.pi, np.pi, size=X_init_abs.shape)
    
#     cn = X_init_abs * np.exp(1j * X_init_phase)
#     _, t0 = scipy.signal.istft(cn, nperseg=1024, noverlap=256)
#     _, _, t0 = scipy.signal.stft(t0, nperseg=1024, noverlap=256)
    
#     for n in range(num_iter):
#         _, tn = scipy.signal.istft(cn, nperseg=1024, noverlap=256)
#         _, _, tn = scipy.signal.stft(tn, nperseg=1024, noverlap=256)
#         X_phase = np.angle(tn)
#         tn = X_init_abs * np.exp(1j * X_phase)
        
#         cn = tn + alpha * (tn - t_n_1)
    
#     _, x = scipy.signal.istft(X)
#     sf.write(out_audio_path, x, samplerate=sr)

#     return x
    
