import os 

import torch
from pathlib import Path
import librosa
import numpy as np
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from dataset import build_dataloader
from utils.utils import save_to_json

melgan_comparisons_dir = config.COMPARISONS_DIR / 'melgan'
metrics_path = melgan_comparisons_dir / 'test_metrics.json'
if not os.path.exists(melgan_comparisons_dir):
    os.mkdir(melgan_comparisons_dir)
    
hparams = config.create_hparams()

test_dl = build_dataloader(hparams, config.DATA_DIR, "audio", "test")

pesq = PerceptualEvaluationSpeechQuality(fs = hparams.sr, mode= "wb")
stoi = ShortTimeObjectiveIntelligibility(fs = hparams.sr)

vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

test_scores = {"stoi": 0.,
                "pesq": 0.} 

pbar = tqdm(test_dl, desc="Evaluation")
with torch.no_grad():
    for n, batch in enumerate(pbar):
        x_wav = batch['wav'].to(hparams.device)
        x_mel = vocoder(x_wav)
        x_wav_hat = vocoder.inverse(x_mel)
        try:
            pesq_metric = pesq(x_wav_hat, x_wav) 
        except:
            pesq_metric = test_scores["pesq"]
        test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))

        stoi_score = stoi(x_wav_hat, x_wav)
        test_scores["stoi"] += ((1./(n+1))*(stoi_score-test_scores["stoi"]))
        
        test_scores = {k: round(float(v), 4) for k, v in test_scores.items() if v != 0.}
        scores_to_print = str(test_scores)
        pbar.set_postfix_str(scores_to_print)

save_to_json(test_scores, metrics_path)
        
        
        