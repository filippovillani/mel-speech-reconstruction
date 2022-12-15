import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from metrics import si_ssnr_metric 

def griffin_lim_librosa(spectrogram: np.ndarray, 
                        n_iter: int = 512, 
                        init: str = "random"):
    
    x = librosa.griffinlim(spectrogram, n_iter=n_iter, init=init)      

    return x


def griffin_lim_base(spectrogram: np.ndarray, 
                     n_iter: int = 512, 
                     n_fft: int = 1024,
                     init: str = "zeros",
                     eval: bool = False):
    
    if init == "zeros":
        X_init_phase = np.zeros(spectrogram.shape)    
    elif init =="random":
        X_init_phase = np.random.uniform(-np.pi, np.pi, size=spectrogram.shape)
    else:
        raise ValueError("init must be 'zeros' or 'random'")
    
    X = spectrogram * np.exp(1j * X_init_phase)
    snr_hist = []
    for n in range(n_iter):
        X_hat = librosa.istft(X, n_fft=n_fft)    # G+ cn
        X_hat = librosa.stft(X_hat, n_fft=n_fft) # G G+ cn  
        X_phase = np.angle(X_hat) 
        X = spectrogram * np.exp(1j * X_phase)   # Pc1(Pc2(cn-1))  
        if eval and n>0:
            snr_hist.append(si_ssnr_metric(spectrogram, np.abs(X_hat)))
    
    x = librosa.istft(X)
    x /= np.max(x)
    
    return x, snr_hist


def fast_griffin_lim(spectrogram: np.ndarray,
                    n_iter: int = 512,
                    alpha: float = 0.99, 
                    n_fft: int = 1024,
                    init: str = "zeros",
                    eval: bool = False):

    if init == "zeros":
        X_init_phase = np.zeros(spectrogram.shape)    
    elif init =="random":
        X_init_phase = np.random.uniform(-np.pi, np.pi, size=spectrogram.shape)
    else:
        raise ValueError("init must be 'zeros' or 'random'")
    
    # Initialize the algorithm
    X = spectrogram * np.exp(1j * X_init_phase)
    prev_proj = librosa.istft(X, n_fft=n_fft)
    prev_proj = librosa.stft(prev_proj, n_fft=n_fft)
    prev_proj_phase = np.angle(prev_proj) 
    prev_proj = spectrogram * np.exp(1j * prev_proj_phase) 
    
    snr_hist = []
    for n in range(n_iter+1):
        curr_proj = librosa.istft(X, n_fft=n_fft)    # G+ cn            
        curr_proj = librosa.stft(curr_proj, n_fft=n_fft) # G G+ cn  
          
        if eval and n>0:
            snr_hist.append(si_ssnr_metric(spectrogram, np.abs(curr_proj)))
        
        curr_proj_phase = np.angle(curr_proj) 
        curr_proj = spectrogram * np.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))  
            
        X = curr_proj + alpha * (curr_proj - prev_proj)
        prev_proj = curr_proj

    x = librosa.istft(X, n_fft=n_fft)
    x /= np.max(np.abs(x))

    return x, snr_hist