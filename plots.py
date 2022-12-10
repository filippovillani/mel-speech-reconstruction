import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fft import fft, fftfreq
import json

import config

def plot_reconstructed_audio(original_audio: np.ndarray, 
                             reconstructed_audio: np.ndarray, 
                             save_path: str, 
                             sr: int = 16000):
    
    out_img_path = config.GLA_RESULTS_DIR / (str(save_path) + '_reconstructed.png')

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
                        save_path: str):
    out_img_path = config.GLA_RESULTS_DIR / (str(save_path) + '_numiter.png')

    plt.figure() 
    plt.semilogx(snr_hist)
    plt.xlabel("n_iter")
    plt.ylabel("SI-SSNR [dB]")
    plt.title(save_path)
    plt.grid()
    
    plt.savefig(out_img_path)

def plot_train_hist(train_state_path: str):

    train_state_path = config.RESULTS_DIR / train_state_path
    with open(train_state_path) as fp:
        training_state = json.load(fp)

    plt.figure()
    plt.plot(range(1, 1+training_state["epochs"]), training_state["train_loss_hist"], label='train loss')
    plt.plot(range(1, 1+training_state["epochs"]), training_state["val_loss_hist"], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('SI-NSR [dB]')
    plt.legend()
    plt.grid()
    plt.savefig('train_hist')