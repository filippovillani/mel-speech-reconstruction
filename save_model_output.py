import matplotlib.pyplot as plt
import torch
import librosa.display
import soundfile as sf
import numpy as np

from model import MelSpec2Spec
from dataset import build_dataloaders
from griffinlim import fast_griffin_lim
import config

weights_path = 'weights/prova01/prova01_weights'
out_path = config.RESULTS_DIR / 'out01.wav'
out_hat_path = config.RESULTS_DIR / 'out01_hat.wav'

hparams = config.create_hparams()

model = MelSpec2Spec(hparams).float().to(config.DEVICE)
model.eval()
model.load_state_dict(torch.load(weights_path))

_, val_dl = build_dataloaders(config.DATA_DIR, hparams)

for n, el in enumerate(val_dl):
    mel = el['melspectr'].float().to(config.DEVICE)
    spectr = el['spectr'].float().to(config.DEVICE)
    
    spec_hat = model(mel)
    mel_hat = model.compute_mel_spectrogram(spec_hat)
    
    out, _ = fast_griffin_lim(np.abs(spectr.cpu().numpy()[0]))
    sf.write(str(out_path), out, samplerate = hparams.sr)
    
    out_hat, _ = fast_griffin_lim(np.abs(spec_hat.cpu().detach().numpy()[0]))
    sf.write(str(out_hat_path), out_hat[0], samplerate = hparams.sr)


    
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.specshow(mel.cpu().numpy()[0], sr=hparams.sr, n_fft=hparams.n_fft, hop_length=hparams.hop_len)
    plt.subplot(2,1,2)
    librosa.display.specshow(mel_hat.cpu().detach().numpy().squeeze()[0], sr=hparams.sr, n_fft=hparams.n_fft, hop_length=hparams.hop_len)
    plt.savefig('results/prova_mel_predicted')
    
    if n == 1:
        break