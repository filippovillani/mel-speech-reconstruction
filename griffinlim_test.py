import torch
from tqdm import tqdm
from pystoi import stoi

from metrics import si_snr_metric
from utils.audioutils import min_max_normalization
from utils.utils import r2_to_mag_phase
from griffinlim import fast_griffin_lim, griffin_lim_base
from dataset import build_dataloader
import config

hparams = config.create_hparams()
test_dl = build_dataloader(hparams, config.DATA_DIR, "test")

stoi_score = 0.
snr_score = 0.
pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
with torch.no_grad():
    for n, batch in enumerate(pbar):
        x_stft = batch['stft'].squeeze()
        x_wav = min_max_normalization(torch.istft(x_stft, n_fft=hparams.n_fft).numpy().squeeze())
        x_wav_hat, _ = fast_griffin_lim(torch.abs(x_stft).numpy(), n_iter=64)

        stoi_metric = stoi(x_wav, x_wav_hat, fs_sig = hparams.sr)
        stoi_score += ((1./(n+1))*(stoi_metric-stoi_score))

        pbar.set_postfix_str(f'stoi: {stoi_score:.5f}, snr: {snr_score:.5f}')  