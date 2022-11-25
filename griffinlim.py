import numpy as np
from pathlib import Path
import soundfile as sf
import scipy.signal

def griffin_lim_base(spectrogram, out_path, num_iter=512):
    X_init_abs = np.abs(spectrogram)
    X_init_phase = np.random.uniform(-np.pi, np.pi, size=X_init_abs.shape)
    X = X_init_abs * np.exp(1j * X_init_phase)
    
    for n in range(num_iter):
        _, X_hat = scipy.signal.istft(X)
        _, _, X_hat = scipy.signal.stft(X_hat)
        X_phase = np.angle(X_hat)
        X = X_init_abs * np.exp(1j * X_phase)
    
    _, x = scipy.signal.istft(X)
    sf.write(str(out_path), x, samplerate=16000)
    
    # plt.figure()
    # plt.plot(x)
    # plt.plot(audio)
    
if __name__ == "__main__":
    num_iter = 1024
    audio_path = Path(r'D:\GitHub_Portfolio\UNet_SpeechEnhancer\mixture_example\clean0.wav')
    in_path = Path(r'D:\GitHub_Portfolio\dsp\in.wav')
    out_path = Path(r'D:\GitHub_Portfolio\dsp\out.wav')
    with open(in_path, 'rb') as f:
        audio, _ = sf.read(f)
    
    _, _, spectrogram = scipy.signal.stft(audio)
    griffin_lim_base(spectrogram, num_iter)