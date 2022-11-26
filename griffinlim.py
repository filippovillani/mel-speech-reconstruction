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
    X_init_phase = np.random.uniform(-np.pi, np.pi, size=X_init_abs.shape)
    X = X_init_abs * np.exp(1j * X_init_phase)
    
    for n in range(num_iter):
        _, X_hat = scipy.signal.istft(X, nperseg=1024, noverlap=256)
        _, _, X_hat = scipy.signal.stft(X_hat, nperseg=1024, noverlap=256)
        X_phase = np.angle(X_hat)
        X = X_init_abs * np.exp(1j * X_phase)
    
    _, x = scipy.signal.istft(X)
    sf.write(out_audio_path, x, samplerate=sr)
    
    return x
    
    
    
