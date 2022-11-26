import numpy as np
import matplotlib.pyplot as plt
import librosa.display

import config

def plot_reconstructed_audio(original_audio: np.ndarray, 
                             reconstructed_audio: np.ndarray, 
                             out_path: str, 
                             sr: int = 16000):
    
    out_img_path = config.RESULTS_DIR / (str(out_path) + '.png')

    plt.figure()
    librosa.display.waveshow(original_audio, sr=sr, label="Original signal")
    librosa.display.waveshow(reconstructed_audio, sr=sr, label="Reconstructed signal")
    plt.grid()
    plt.legend()
    plt.savefig(out_img_path)