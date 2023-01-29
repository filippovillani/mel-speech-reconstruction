import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import json
from argparse import Namespace



def plot_train_hist(experiment_dir):
    """
        Plots loss and metric histories for training and validation sets
    Args:
        experiment_name (str): train_state.json directory
    """
    """
        Plots loss and metric histories for training and validation sets
    Args:
        experiment_name (str): train_state.json directory
    """
    train_state_path = experiment_dir / 'train_state.json'
    experiment_name = experiment_dir.stem
    with open(train_state_path) as fp:
        training_state = json.load(fp)

    for metric in training_state["train_hist"].keys():
        save_path = experiment_dir / (metric + '.png')
        plt.figure()
        plt.plot(range(1, 1+training_state["epochs"]), training_state["train_hist"][metric], label='train')
        plt.plot(range(1, 1+training_state["epochs"]), training_state["val_hist"][metric], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(experiment_name)
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.close()

def plot_prediction(mel: np.ndarray,
                    mel_hat: np.ndarray,
                    hparams: Namespace,
                    save_path: str):
    
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
    plt.title('STFT-spectrogram predicted')
    plt.colorbar(format="%+2.f dB")
    plt.savefig(save_path)