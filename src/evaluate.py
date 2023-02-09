import argparse
import os

import librosa
import torch
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

import config
from dataset import build_dataloader
from networks.build_model import build_model
from griffinlim import griffin_lim, fast_griffin_lim
from utils.audioutils import to_db, to_linear, compute_wav, normalize_db_spectr, denormalize_db_spectr
from utils.utils import save_to_json


class Tester:
    def __init__(self, args):
        
        self.model_name = args.model_name
        self.task = args.task
        self._set_paths(args.experiment_name)
        
        if self.model_name in ["gla", "fgla"]: # this is for gla and fgla
            self.hprms = config.create_hparams()
            self.gla_iter = args.num_iter
        elif self.model_name in ["degli", "pinvconv"]: 
            self.hprms = config.load_config(self.config_path)
            self.hprms.batch_size = 1
            self.model = build_model(self.hprms, args.model_name, self.experiment_weights_dir, best_weights=True)
            self.model.repetitions = args.degli_blocks
            self.model.eval()
        elif self.model_name == "pinv":
            self.hprms = config.create_hparams()
            self.model = build_model(self.hprms, args.model_name)
            self.model.eval()
        else:
            raise ValueError(f'model_name must be one of ["pinvconv", "degli", "gla", "fgla"], \
                                    received: {args.model_name}')
                    
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = self.hprms.sr, 
                                                         n_fft = self.hprms.n_fft, 
                                                         n_mels = self.hprms.n_mels)).to(self.hprms.device)
        
        self.pesq = PerceptualEvaluationSpeechQuality(fs = self.hprms.sr, mode= "wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs = self.hprms.sr)
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(self.hprms.device)
        
    def test_model(self, 
                   test_dl: DataLoader):
        
        
        test_scores = {"stoi": 0.,
                       "pesq": 0.,
                       "si-sdr": 0.} 
        
        pbar = tqdm(test_dl, desc="Evaluation")
        with torch.no_grad():
            for n, batch in enumerate(pbar):
                if self.task == "melspec2spec":
                    x_stftspec_db_norm, x_melspec_db_norm = self._preprocess_mel2spec_batch(batch)
                    x_stftspec_hat_db_norm = self.model(x_melspec_db_norm).squeeze(1)
                    sdr_metric = self.si_sdr(to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm)),
                                             to_linear(denormalize_db_spectr(x_stftspec_db_norm)))
                    test_scores["si-sdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-sdr"]))  
                    
                elif self.task == "spec2wav":
                    x_stft = batch["stft"].to(self.hprms.device)
                    x_stft_mag = torch.abs(x_stft).float()
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    
                    if self.model_name == "degli":
                        x_stft_noisy = self._initialize_random_phase(x_stft_mag)
                        x_stft_hat = self.model(x_stft_noisy, x_stft_mag)
                        x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    elif self.model_name == "gla":
                        x_wav_hat = griffin_lim(x_stft_mag, n_iter=self.gla_iter).squeeze()
                    elif self.model_name == "fgla":
                        x_wav_hat = fast_griffin_lim(x_stft_mag, n_iter=self.gla_iter).squeeze()
                    
                    # Compute metrics:
                    stoi_metric = self.stoi(x_wav_hat, x_wav) 
                    test_scores["stoi"] += ((1./(n+1))*(stoi_metric-test_scores["stoi"]))
                        
                    try:
                        pesq_metric = self.pesq(x_wav_hat, x_wav) 
                    except:
                        pesq_metric = test_scores["pesq"]
                    test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))
                                    
                test_scores = {k: round(float(v), 4) for k, v in test_scores.items() if v != 0.}
                scores_to_print = str(test_scores)
                pbar.set_postfix_str(scores_to_print)
                
                # if n == 50:
                #     break
                
        save_to_json(test_scores, self.test_metrics_path)    

    def _preprocess_mel2spec_batch(self, batch):
        
        x_stft = batch["stft"].to(self.hprms.device)
        x_stftspec = torch.abs(x_stft).float()
        x_melspec = torch.matmul(self.melfb, x_stftspec**2).unsqueeze(1)
        
        x_stftspec_db_norm = normalize_db_spectr(to_db(x_stftspec))
        x_melspec_db_norm = normalize_db_spectr(to_db(x_melspec, power_spectr=True))
        
        return x_stftspec_db_norm, x_melspec_db_norm
    
         
    def _initialize_random_phase(self, x_stft_mag):
        
        phase = torch.zeros_like(x_stft_mag)
        x_stft = x_stft_mag * torch.exp(1j * phase)
        return x_stft        


    def _set_paths(self, experiment_name):
        
        if args.task == "melspec2spec":
            results_dir = config.MELSPEC2SPEC_DIR
        elif args.task == "spec2wav":
            results_dir = config.SPEC2WAV_DIR
        else:
            raise ValueError(f"task must be one of ['melspec2spec', 'spec2wav'], received {args.task}")        
        
        experiment_dir = results_dir / experiment_name            
        
        self.experiment_weights_dir = config.WEIGHTS_DIR / experiment_name
        self.test_metrics_path = experiment_dir / 'test_metrics.json'
        self.config_path = experiment_dir / "config.json"
        
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)

    
def main(args):
 
    tester = Tester(args)    
    test_dl = build_dataloader(tester.hprms, config.DATA_DIR, "test")
    _ = tester.test_model(test_dl) 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task',
                        type=str,
                        choices=["melspec2spec", "spec2wav"],
                        default='spec2wav')
    
    parser.add_argument('--model_name',
                        choices = ["pinvconv", "pinvunet", "pinv", "fgla", "gla", "degli"],
                        type=str,
                        default = 'degli')
    
    parser.add_argument('--experiment_name',
                        type=str,
                        default='degli_B1_deglidata_fromnoiseM6P12')
    
    parser.add_argument('--degli_blocks',
                        type=int,
                        default=100)
    
    parser.add_argument('--num_iter',
                        type=int,
                        default=1000)

    args = parser.parse_args()
    main(args)