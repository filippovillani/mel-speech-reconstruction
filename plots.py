import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.fft import fft, fftfreq
import json
from argparse import Namespace

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
    plt.ylabel("SI-SNR [dB]")
    plt.title(save_path)
    plt.grid()
    
    plt.savefig(out_img_path)

def plot_train_hist(experiment_name: str):
    """
        Plots loss and metric histories for training and validation sets
    Args:
        experiment_name (str): train_state.json directory
    """
    experiment_dir = config.MELSPEC2SPEC_DIR / experiment_name
    train_state_path = experiment_dir / 'train_state.json'
    loss_img_path = experiment_dir / 'loss_hist.png'
    metric_img_path = experiment_dir / 'metric_hist.png'
    
    with open(train_state_path) as fp:
        training_state = json.load(fp)

    plt.figure()
    plt.plot(range(1, 1+training_state["epochs"]), training_state["train_loss_hist"], label='train loss')
    plt.plot(range(1, 1+training_state["epochs"]), training_state["val_loss_hist"], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title(experiment_name)
    plt.legend()
    plt.grid()
    plt.savefig(loss_img_path)

    plt.figure()
    plt.plot(range(1, 1+training_state["epochs"]), training_state["train_score_hist"], label='train metric')
    plt.plot(range(1, 1+training_state["epochs"]), training_state["val_score_hist"], label='val metric')
    plt.xlabel('Epochs')
    plt.ylabel('SI-SNR [dB]')
    plt.title(experiment_name)
    plt.legend()
    plt.grid()
    plt.savefig(metric_img_path)

def plot_prediction(mel: np.ndarray,
                    mel_hat: np.ndarray,
                    hparams: Namespace,
                    experiment_name: str):
    
    out_path = config.MELSPEC2SPEC_DIR / experiment_name / 'prediction.png'
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.specshow(mel, 
                             sr=hparams.sr, 
                             n_fft=hparams.n_fft, 
                             hop_length=hparams.hop_len, 
                             x_axis='time', 
                             y_axis='hz')
    plt.title('STFT-spectrogram')
    plt.colorbar(format="%+2.f dB")
    
    plt.subplot(2,1,2)
    librosa.display.specshow(mel_hat, 
                             sr=hparams.sr, 
                             n_fft=hparams.n_fft, 
                             hop_length=hparams.hop_len, 
                             x_axis='time', 
                             y_axis='hz')
    plt.title('STFT-spectrogram predicted')
    plt.colorbar(format="%+2.f dB")
    plt.savefig(out_path)