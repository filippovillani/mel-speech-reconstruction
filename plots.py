import matplotlib.pyplot as plt
import librosa.display

import config

def plot_reconstructed_audio(original_audio, reconstructed_audio, out_path, sr=16000):
    out_img_path = config.RESULTS_DIR / (str(out_path) + '.png')

    plt.figure()
    librosa.display.waveshow(original_audio, sr=sr, label="Original signal")
    librosa.display.waveshow(reconstructed_audio, sr=sr, label="Reconstructed signal")
    plt.grid()
    plt.legend()
    plt.savefig(out_img_path)