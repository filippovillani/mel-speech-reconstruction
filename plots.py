import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fft import fft, fftfreq

import config

def plot_reconstructed_audio(original_audio: np.ndarray, 
                             reconstructed_audio: np.ndarray, 
                             save_path: str, 
                             sr: int = 16000):
    
    out_img_path = config.RESULTS_DIR / (str(save_path) + '.png')

    plt.figure()
    librosa.display.waveshow(original_audio, sr=sr, label="Original signal")
    librosa.display.waveshow(reconstructed_audio, sr=sr, label="Reconstructed signal")
    plt.grid()
    plt.legend()
    plt.savefig(out_img_path)
    
def plot_window(window: np.ndarray,
                spectrum: np.ndarray, 
                save_path: str,
                N: int,
                T: int):
    
    out_img_path = config.WINDOWS_IMG_DIR / (str(save_path) + '.png')

    xf = fftfreq(N, T)[:N//2]
    spectrum = 20*np.log10(2/N *abs(spectrum[0:N//2]))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(window)
    plt.xlabel('Time')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(xf, spectrum)
    plt.xlabel('Frequency')
    plt.grid()
    
    plt.savefig(out_img_path)
    
def plot_metric_numiter(snr_hist: list, 
                        save_path: str,
                        metric: str = "SI-SNR"):
    out_img_path = config.RESULTS_DIR / (str(save_path) + metric + '_numiter.png')

    plt.figure() 
    plt.plot(snr_hist)
    plt.xlabel("n_iter")
    plt.ylabel(metric)
    plt.title(save_path)
    plt.grid()
    
    plt.savefig(out_img_path)
