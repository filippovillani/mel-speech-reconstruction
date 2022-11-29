import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from metrics import si_ssnr 
import config

def griffin_lim_librosa(spectrogram: np.ndarray, 
                        out_path: str, 
                        num_iter: int = 512, 
                        sr: int = 16000):
    
    out_audio_path = config.RESULTS_DIR / (str(out_path) + '.wav')
    x = librosa.griffinlim(spectrogram, n_iter=num_iter)      
    sf.write(out_audio_path, x, samplerate=sr)

    return x


def griffin_lim_base(spectrogram: np.ndarray, 
                     out_path: str, 
                     num_iter: int = 512, 
                     n_fft: int = 1024,
                     sr: int = 16000,
                     init: str = "random",
                     eval: bool = True):
    
    out_audio_path = config.RESULTS_DIR / (str(out_path) + '.wav')
    if init == "zeros":
        X_init_phase = np.zeros(spectrogram.shape)    
    elif init =="random":
        X_init_phase = np.random.uniform(-np.pi, np.pi, size=spectrogram.shape)
    else:
        raise ValueError("init must be 'zeros' or 'random'")
    
    X = spectrogram * np.exp(1j * X_init_phase)
    snr_hist = []
    for _ in range(num_iter):
        X_hat = librosa.istft(X, n_fft=n_fft)    # G+ cn
        X_hat = librosa.stft(X_hat, n_fft=n_fft) # G G+ cn  
        X_phase = np.angle(X_hat) 
        X = spectrogram * np.exp(1j * X_phase)   # Pc1(Pc2(cn-1))  
        if eval:
            snr_hist.append(si_ssnr(spectrogram, np.abs(X)))
    x = librosa.istft(X)
    sf.write(out_audio_path, x, samplerate=sr)
    
    return x, snr_hist

    
    
def fast_griffin_lim(spectrogram: np.ndarray,
                    out_path: str, 
                    num_iter: int = 512,
                    alpha: float = 0.99, 
                    n_fft: int = 1024,
                    sr: int = 16000,
                    init: str = "random",
                    eval: bool = True):

    out_audio_path = config.RESULTS_DIR / (str(out_path) + '.wav')
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
    for _ in range(num_iter):
        curr_proj = librosa.istft(X, n_fft=n_fft)    # G+ cn            
        curr_proj = librosa.stft(curr_proj, n_fft=n_fft) # G G+ cn    

        curr_proj_phase = np.angle(curr_proj) 
        curr_proj = spectrogram * np.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))  
            
        X = curr_proj + alpha * (curr_proj - prev_proj)
        prev_proj = curr_proj
        if eval:
            snr_hist.append(si_ssnr(spectrogram, np.abs(X)))
               
    x = librosa.istft(X, n_fft=n_fft)
    sf.write(out_audio_path, x, samplerate=sr)

    return x, snr_hist
        