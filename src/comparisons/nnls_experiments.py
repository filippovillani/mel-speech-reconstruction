import librosa
import librosa.display
import torch
import matplotlib.pyplot as plt
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from griffinlim import fast_griffin_lim
from utils.audioutils import (to_linear, denormalize_db_spectr, 
                              open_audio, normalize_db_spectr, to_db)
from utils.utils import save_to_json
from metrics import SI_SSDR
from dataset import build_dataloader

audio_example = config.DATA_DIR / 'in.wav'
score_path = config.COMPARISONS_DIR / 'nnls_score.json'
nnls_img_path = config.COMPARISONS_DIR / 'nnls_spectra.png'

hparams = config.create_hparams()
test_dl = build_dataloader(hparams, config.DATA_DIR, "spec2wav", ds_type="test")
sissdr = SI_SSDR()
pesq = PerceptualEvaluationSpeechQuality(fs = hparams.sr, mode="wb")
stoi = ShortTimeObjectiveIntelligibility(fs=hparams.sr)
melfb = torch.as_tensor(librosa.filters.mel(sr = hparams.sr, 
                                            n_fft = hparams.n_fft, 
                                            n_mels = hparams.n_mels))
score = {"si-ssdr": 0.,
         "stoi": 0.,
         "pesq": 0.,}

pbar = tqdm(test_dl, postfix='[]')
for n, batch in enumerate(pbar):
    x_stft = batch["stft"]
    x_wav = torch.istft(x_stft, n_fft=hparams.n_fft).squeeze()
    x_stftspec_db = normalize_db_spectr(to_db(torch.abs(x_stft))).float()
    x_stft = to_linear(denormalize_db_spectr(x_stftspec_db))
    x_melspec_db = torch.matmul(melfb, x_stftspec_db)
    x_melspec = to_linear(denormalize_db_spectr(x_melspec_db/torch.max(x_melspec_db))).squeeze()
    x_stft_hat = torch.as_tensor(librosa.feature.inverse.mel_to_stft(x_melspec.numpy(), sr=hparams.sr, n_fft=hparams.n_fft))
    
    x_wav_hat = fast_griffin_lim(x_stft_hat, n_fft=hparams.n_fft, n_iter = 200)
    sdr_ = sissdr(x_stft_hat, x_stft)
    stoi_ = stoi(x_wav_hat, x_wav)
    pesq_ = pesq(x_wav_hat, x_wav)
    
    score["si-ssdr"] += ((1./(n+1))*(sdr_-score["si-ssdr"])).item()
    score["stoi"] += ((1./(n+1))*(stoi_-score["stoi"])).item()
    score["pesq"] += ((1./(n+1))*(pesq_-score["pesq"])).item()
    
    scores_to_print = str({k: round(float(v), 4) for k, v in score.items()})
    pbar.set_postfix_str(scores_to_print)
save_to_json(score, score_path) 

x_ex_wav = open_audio(audio_example, sr=hparams.sr)
x_ex_stftspec = torch.abs(torch.stft(input=x_ex_wav, 
                                     n_fft=hparams.n_fft,
                                     hop_length=hparams.hop_len,
                                     window = torch.hann_window(hparams.n_fft),
                                     return_complex=True))
x_ex_stftspec_db = normalize_db_spectr(to_db(x_ex_stftspec))
x_ex_melspec_db = torch.matmul(melfb, x_ex_stftspec_db)
x_ex_melspec = to_linear(denormalize_db_spectr(x_ex_melspec_db/torch.max(x_ex_melspec_db))).squeeze()
x_ex_stftspec_hat = torch.as_tensor(librosa.feature.inverse.mel_to_stft(x_ex_melspec.numpy(), sr=hparams.sr, n_fft=hparams.n_fft))

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
img = librosa.display.specshow(denormalize_db_spectr(x_ex_stftspec_db).squeeze().numpy(), 
                               sr=hparams.sr, 
                               n_fft=hparams.n_fft, 
                               x_axis='time', 
                               y_axis='hz',
                               ax=ax[0])
ax[0].set(title='|X_stft|')
ax[0].label_outer()

librosa.display.specshow(to_db(x_ex_stftspec_hat).squeeze().numpy(), 
                         sr=hparams.sr, 
                         n_fft=hparams.n_fft, 
                         x_axis='time',
                         y_axis='hz',
                         ax=ax[1])
ax[1].set(title='|X_stft_hat|')
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.savefig(nnls_img_path)