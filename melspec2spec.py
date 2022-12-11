import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

from griffinlim import fast_griffin_lim
import config


audio_path = config.DATA_DIR / 'in.wav'
x_path = config.MELSPEC2SPEC_DIR / 'x.wav'
x_hat_path = config.MELSPEC2SPEC_DIR / 'x_hat.wav'
img_path = config.MELSPEC2SPEC_DIR / 'spectr.png'

with open(audio_path, 'rb') as f:
    audio, sr = sf.read(f)

audio = (audio - np.mean(audio)) / np.std(audio)

melfb = librosa.filters.mel(sr=16000, n_mels=96, n_fft=1024)
spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024)),
                               ref=np.max)
melspec = np.dot(melfb, spec)
spec_hat = np.dot(np.linalg.pinv(melfb), melspec)
    
x = fast_griffin_lim(librosa.db_to_amplitude(spec))
x_hat = fast_griffin_lim(librosa.db_to_amplitude(spec_hat))

sf.write(str(x_path), x, samplerate=16000)
sf.write(str(x_hat_path), x_hat, samplerate=16000)

plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(spec, sr=16000, n_fft=1024, hop_length=256, win_length=1024)
plt.subplot(2,1,2)
librosa.display.specshow(spec_hat, sr=16000, n_fft=1024, hop_length=256, win_length=1024)
plt.savefig(img_path)
