import argparse
import os

import librosa
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from metrics import SI_SDR
from dataset import build_dataloader
from networks.build_model import build_model
from griffinlim import griffin_lim, fast_griffin_lim
from utils.audioutils import (to_db, to_linear, compute_wav, initialize_random_phase,
                              normalize_db_spectr, denormalize_db_spectr)
from utils.utils import save_to_json, load_config


class Tester:
    def __init__(self, args):
        
        self.spec2wav_model_name = args.spec2wav_model_name
        self.melspec2spec_model_name = args.melspec2spec_model_name
        self.spec2wav_weights_path = args.spec2wav_weights_path
        self.melspec2spec_weights_path = args.melspec2spec_weights_path
        self.experiment_name = args.experiment_name
        self.task = args.task
        self._set_paths()
        
        if self.task in ["melspec2wav", "spec2wav"]:
            if self.spec2wav_model_name in ["gla", "fgla"]: # this is for gla and fgla
                self.hprms = config.create_hparams()
                self.gla_iter = args.num_iter
            else: 
                self.hprms = load_config(self.config_path)
                self.spec2wav_model = build_model(self.hprms, self.spec2wav_model_name, self.spec2wav_weights_dir, best_weights=True)
                self.spec2wav_model.repetitions = args.degli_blocks
                self.spec2wav_model.eval()
                
        if self.task in ["melspec2wav", "melspec2spec"]:
            if self.melspec2spec_model_name == "pinv":
                self.hprms = config.create_hparams()
                self.melspec2spec_model = build_model(self.hprms, args.melspec2spec_model_name)
                self.melspec2spec_model.eval()
            else:
                self.hprms = load_config(self.config_path)
                self.melspec2spec_model = build_model(self.hprms, args.melspec2spec_model_name, self.melspec2spec_weights_dir, best_weights=True)
                self.melspec2spec_model.eval()
               
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = self.hprms.sr, 
                                                         n_fft = self.hprms.n_fft, 
                                                         n_mels = self.hprms.n_mels)).to(self.hprms.device)
        
        self.pesq = PerceptualEvaluationSpeechQuality(fs = self.hprms.sr, mode= "wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs = self.hprms.sr)
        self.sisdr = SI_SDR().to(self.hprms.device)
        
    def test_model(self, 
                   test_dl: DataLoader):
        
        
        test_scores = {"stoi": 0.,
                       "pesq": 0.,
                       "si-sdr": 0.} 
        
        pbar = tqdm(test_dl, desc="Evaluation")
        with torch.no_grad():
            for n, batch in enumerate(pbar):
                if self.task == "melspec2spec":
                    x_stftspec, x_melspec_db_norm = self._preprocess_melspec2spec_batch(batch)
                    
                    x_stftspec_hat_db_norm = self.melspec2spec_model(x_melspec_db_norm).squeeze(1)
                    x_stftspec_hat = to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm))  
                    sdr_metric = self.sisdr(x_stftspec_hat, x_stftspec)
                    test_scores["si-sdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-sdr"]))
                    
                elif self.task == "spec2wav":
                    x_stft = batch["stft"].to(self.hprms.device)
                    x_stftspec = torch.abs(x_stft).float()
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    
                    if self.spec2wav_model_name == "degli":
                        x_stft_noisy = initialize_random_phase(x_stftspec)
                        x_stft_hat = self.spec2wav_model(x_stft_noisy, x_stftspec)
                        x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    elif self.spec2wav_model_name == "gla":
                        x_wav_hat = griffin_lim(x_stftspec, n_iter=self.gla_iter).squeeze()
                    elif self.spec2wav_model_name == "fgla":
                        x_wav_hat = fast_griffin_lim(x_stftspec, n_iter=self.gla_iter).squeeze()
                    
                    # Compute metrics:
                    stoi_metric = self.stoi(x_wav_hat, x_wav) 
                    test_scores["stoi"] += ((1./(n+1))*(stoi_metric-test_scores["stoi"]))
                        
                    try:
                        pesq_metric = self.pesq(x_wav_hat, x_wav) 
                    except:
                        pesq_metric = test_scores["pesq"]
                    test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))
                                    
                elif self.task == "melspec2wav":
                    x_stftspec, x_melspec_db_norm, x_wav = self._preprocess_melspec2wav_batch(batch)
                    
                    x_stftspec_hat_db_norm = self.melspec2spec_model(x_melspec_db_norm).squeeze(1)
                    x_stftspec_hat = to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm))  
                    
                    if self.spec2wav_model_name == "degli":
                        x_stft_noisy = initialize_random_phase(x_stftspec_hat)
                        x_stft_hat = self.spec2wav_model(x_stft_noisy, x_stftspec_hat)
                        x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    elif self.spec2wav_model_name == "gla":
                        x_wav_hat = griffin_lim(x_stftspec_hat, n_iter=self.gla_iter).squeeze()
                    elif self.spec2wav_model_name == "fgla":
                        x_wav_hat = fast_griffin_lim(x_stftspec_hat, n_iter=self.gla_iter).squeeze()
                    
                    sdr_metric = self.sisdr(x_stftspec_hat, x_stftspec)
                    stoi_metric = self.stoi(x_wav_hat, x_wav)
                    pesq_metric = self.pesq(x_wav_hat, x_wav)
                    test_scores["si-sdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-sdr"]))
                    test_scores["stoi"] += ((1./(n+1))*(stoi_metric-test_scores["stoi"]))
                    test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))
                    
                
                test_scores = {k: round(float(v), 4) for k, v in test_scores.items() if v != 0.}
                scores_to_print = str(test_scores)
                pbar.set_postfix_str(scores_to_print)
                

        save_to_json(test_scores, self.test_metrics_path)    

    def _preprocess_melspec2spec_batch(self, batch):
        x_stftspec_db_norm = batch["spectrogram"].float().to(self.hprms.device)
        x_stftspec = to_linear(denormalize_db_spectr(x_stftspec_db_norm))  
        x_melspec_db_norm = torch.matmul(self.melfb, x_stftspec_db_norm).unsqueeze(1)
        
        return x_stftspec, x_melspec_db_norm
    

    def _preprocess_melspec2wav_batch(self, batch):

        x_stft = batch["stft"]
        x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
        x_stftspec_db_norm = normalize_db_spectr(to_db(torch.abs(x_stft))).float().to(self.hprms.device)
        x_melspec_db_norm = torch.matmul(self.melfb, x_stftspec_db_norm).unsqueeze(1)
        x_stftspec = to_linear(denormalize_db_spectr(x_stftspec_db_norm))
        return x_stftspec, x_melspec_db_norm, x_wav       

    def _set_paths(self):
        
        if self.task == "melspec2spec":
            results_dir = config.MELSPEC2SPEC_DIR
        elif self.task == "spec2wav":
            results_dir = config.SPEC2WAV_DIR
        elif self.task == "melspec2wav":
            results_dir = config.MELSPEC2WAV_DIR
        else:
            raise ValueError(f"task must be one of ['melspec2spec', 'spec2wav', 'melspec2wav'], received {self.task}")        
        
        experiment_dir = results_dir / self.experiment_name            
        
        self.melspec2spec_weights_dir = config.WEIGHTS_DIR / self.melspec2spec_weights_path
        self.spec2wav_weights_dir = config.WEIGHTS_DIR / self.spec2wav_weights_path
        self.test_metrics_path = experiment_dir / 'test_metrics.json'
        
        if self.task in ["melspec2spec", "spec2wav"]:
            self.config_path = experiment_dir / "config.json"
        else:
            self.config_path = config.MELSPEC2SPEC_DIR / self.melspec2spec_weights_path / "config.json"
        
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

    
def main(args):
 
    tester = Tester(args)    
    test_dl = build_dataloader(tester.hprms, config.DATA_DIR, args.task, "test")
    _ = tester.test_model(test_dl) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--experiment_name',
                        type=str,
                        default='degli_def01')
    
    parser.add_argument('--task',
                        type=str,
                        choices=["melspec2spec", "spec2wav", "melspec2wav"],
                        default='spec2wav')
    
    parser.add_argument('--spec2wav_model_name',
                        choices = ["fgla", "gla", "degli"],
                        type=str,
                        default = 'degli')
    
    parser.add_argument('--spec2wav_weights_path',
                        type=str,
                        default = 'degli_def01')

    parser.add_argument('--melspec2spec_model_name',
                        choices = ["pinvconv", "pinvconvskip", "pinvunet", "pinvconvskip", 
                                   "pinvconvskipnobottleneck", "pinvconvres", "pinv"],
                        type=str,
                        default = 'pinvconvskip')
    
    parser.add_argument('--melspec2spec_weights_path',
                        type=str,
                        default = 'pinvconvres04')
    
    parser.add_argument('--degli_blocks',
                        type=int,
                        default=100)
    
    parser.add_argument('--num_iter',
                        type=int,
                        default=200)

    args = parser.parse_args()
    main(args)