from pathlib import Path
import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import load_json

def plot_train_hist(experiment_dir: Path):
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
    training_state = load_json(train_state_path)

    for metric in training_state["train_hist"].keys():
        save_path = experiment_dir / (metric + '.png')
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, 1+training_state["epochs"]), training_state["train_hist"][metric], label='train')
        plt.plot(range(1, 1+training_state["epochs"]), training_state["val_hist"][metric], label='validation')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(experiment_name)
        plt.legend()
        plt.grid()
        
        plt.savefig(save_path)
        plt.close()


def plot_melspec_prediction(spectrogram: np.ndarray,
                            spectrogram_hat: np.ndarray,
                            sr: int,
                            n_fft: int,
                            hop_len: int,
                            save_path: str):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    img = librosa.display.specshow(spectrogram, 
                                   sr=sr, 
                                   n_fft=n_fft, 
                                   hop_length=hop_len, 
                                   x_axis='time', 
                                   y_axis='hz',
                                   ax=ax[0])
    ax[0].set(title='STFT-spectrogram')
    ax[0].label_outer()
    
    librosa.display.specshow(spectrogram_hat, 
                             sr=sr, 
                             n_fft=n_fft, 
                             hop_length=hop_len, 
                             x_axis='time', 
                             y_axis='hz',
                             ax=ax[1])
    ax[1].set(title='STFT-spectrogram predicted')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(save_path)

def plot_train_hist_degli(experiment_dir: Path):
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
    training_state = load_json(train_state_path)

    for metric in training_state["train_hist"].keys():
        # Train
        save_path = experiment_dir / (metric + '_train.png')
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, 1+training_state["epochs"]), 
                 training_state["train_hist"][metric], 
                 label='train', 
                 linewidth=2,
                 color='blue')
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.title(experiment_name)
        plt.legend()
        plt.grid()
        
        plt.savefig(save_path)
        plt.close()
        
        # Validation is computed differently than training, i better save it in another plot
        if metric != "loss":
            save_path = experiment_dir / (metric + '_validation.png')
            plt.figure(figsize=(10, 8))
            plt.plot(range(1, 1+training_state["epochs"]), 
                     training_state["val_hist"][metric], 
                     label='validation', 
                     linewidth=2,
                     color='red')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.title(experiment_name)
            plt.legend()
            plt.grid()
            
            plt.savefig(save_path)
            plt.close()
   
    
def plot_gla_metrics(metrics: dict,
                     save_path: Path):
    
    n_iter_ax = [n * 10 for n in range(len(metrics["pesq_hist"]))]
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(n_iter_ax, metrics["pesq_hist"], color='b')
    plt.ylabel("WB-PESQ")
    plt.grid()
    plt.title("Fast Griffin-Lim metrics", fontsize=16)
    
    plt.subplot(2,1,2)
    plt.plot(n_iter_ax, metrics["stoi_hist"], color='r')
    plt.xlabel('Number of GLA iterations')
    plt.ylabel("STOI")
    plt.grid()
    
    plt.savefig(save_path)
    plt.close()
    
    
def plot_gla_time(times: list, 
                  save_path: Path):
    
    plt.figure()
    plt.plot(range(1,len(times)+1), times, color='b')
    plt.xlabel("Number of GLA iterations")
    plt.ylabel("Time [s]")
    plt.grid()
    plt.title("Fast Griffin-Lim time per iteration", fontsize=16)
    
    plt.savefig(save_path)
    plt.close()
    
    
def plot_degli_metrics(metrics: dict,
                       save_path: Path):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.plot(range(1,len(metrics["pesq_hist"])+1), metrics["pesq_hist"], color='b')
    plt.ylabel("WB-PESQ")
    plt.grid()
    plt.title("DeGLI metrics", fontsize=16)
    
    plt.subplot(2,1,2)
    plt.plot(range(1,len(metrics["stoi_hist"])+1), metrics["stoi_hist"], color='r')
    plt.xlabel('Number of DeGLI blocks')
    plt.ylabel("STOI")
    plt.grid()
    
    plt.savefig(save_path)
    plt.close()


def plot_degli_time(times: list, 
                    save_path: Path):
    
    plt.figure()
    plt.plot(range(1,len(times)+1), times, color='b')
    plt.xlabel("Number of DeGLI blocks")
    plt.ylabel("Time [s]")
    plt.grid()
    plt.title("Deep Griffin-Lim time per block", fontsize=16)
    plt.savefig(save_path)
    plt.close()

def plot_degli_gla_metrics_time(comparisons_dir: Path,
                                gla_metrics_hist: dict, 
                                gla_time_hist: list,
                                fgla_metrics_hist: dict,
                                fgla_time_hist: list,
                                degli_metrics_hist: dict, 
                                degli_time_hist: list):
    
    save_dir = comparisons_dir / f"gla{len(gla_time_hist)}_degli{len(degli_time_hist)}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    pesq_path = save_dir / "pesqstoi_time.png"
    gla_time_hist = [gla_time_hist[m] for m in range(9,len(gla_time_hist),10)]
    fgla_time_hist = [fgla_time_hist[m] for m in range(9,len(fgla_time_hist),10)]
    
    
    plt.figure(figsize=(8, 3.5))
    plt.subplot(1,2,1)
    plt.plot(gla_time_hist, gla_metrics_hist["stoi_hist"], color='b', label="GLA")
    plt.plot(gla_time_hist, fgla_metrics_hist["stoi_hist"], color='g', label="FGLA")
    plt.plot(degli_time_hist, degli_metrics_hist["stoi_hist"], color='r', label="DeGLI")
    plt.xlabel("Time [s]")
    plt.ylabel("STOI")
    plt.legend()
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(gla_time_hist, gla_metrics_hist["pesq_hist"], color='b', label="GLA")
    plt.plot(fgla_time_hist, fgla_metrics_hist["pesq_hist"], color='g', label="FGLA")
    plt.plot(degli_time_hist, degli_metrics_hist["pesq_hist"], color='r', label="DeGLI")
    plt.xlabel("Time [s]")
    plt.ylabel("PESQ")
    plt.legend()
    plt.grid()
    plt.savefig(pesq_path)
    plt.close()