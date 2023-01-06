import argparse
import json
import os
from time import time

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from dataset import build_dataloader
from metrics import mse, si_sdr_metric
from networks.build_model import build_model
from utils.audioutils import (compute_wav, denormalize_db_spectr,
                              normalize_db_spectr, to_db, to_linear)
from utils.plots import plot_train_hist
from utils.utils import c_to_r2, r2_to_c


class Trainer:
    def __init__(self, args):
        
        self._set_paths(args.experiment_name, args.task)
        self._set_hparams(args.resume_training)
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = self.hprms.sr, 
                                                         n_fft = self.hprms.n_fft, 
                                                         n_mels = self.hprms.n_mels)).to(self.hprms.device)

        self.loss = torch.nn.L1Loss()
        # self.loss = torch.nn.MSELoss()
        self.pesq = PerceptualEvaluationSpeechQuality(fs = self.hprms.sr, 
                                                      mode= "wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs = self.hprms.sr)

        if args.resume_training:
            # Load training state
            with open(self.training_state_path, "r") as fp:
                self.training_state = json.load(fp)
        
            # Load model's weights and optimizer from checkpoint  
            self.model = build_model(self.hprms, args.model_name, self.experiment_weights_dir, best_weights = False)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)        
            self.optimizer.load_state_dict(torch.load(self.ckpt_opt_path)) 
            self.lr_sched = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)
            self.lr_sched.load_statedict(torch.load(self.ckpt_sched_path))
        else:        
            self.model = build_model(self.hprms, args.model_name)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)
            self.lr_sched = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)
 
            self.training_state = {"epochs": 0,
                                   "patience_epochs": 0,  
                                   "best_epoch": 0,
                                   "best_val_loss": 9999,
                                   "best_val_score": 0,
                                   "train_loss_hist": [],
                                   "train_score_hist": [],
                                   "val_loss_hist": [],
                                   "val_score_hist": []} 
               
    
    def train(self, train_dl, val_dl):
        
        print('_____________________________')
        print('       Training start')
        print('_____________________________')
        
        while self.training_state["patience_epochs"] < self.hprms.patience and self.training_state["epochs"] < self.hprms.epochs:
            
            self.training_state["epochs"] += 1 
            print(f'\n§ Train Epoch: {self.training_state["epochs"]}\n')
            
            self.model.train()
            train_loss = 0.
            train_sdr_score = 0.
            train_stoi_score = 0.
            train_pesq_score = 0.
            start_epoch = time()        
            pbar = tqdm(train_dl, desc=f'Epoch {self.training_state["epochs"]}', postfix='[]')
            
            for n, batch in enumerate(pbar):   
                self.optimizer.zero_grad()  
                x_stft = batch["stft"].to(self.hprms.device)
                
                if args.task == "melspec2spec":
                    x_stftspec_db_norm, x_melspec_db_norm = self._preprocess_mel2spec(x_stft)
                    x_stftspec_hat_db_norm = self.model(x_melspec_db_norm).squeeze()
                    
                    loss = mse(x_stftspec_db_norm, x_stftspec_hat_db_norm)
                    train_loss += ((1./(n+1))*(loss-train_loss))
                    loss.backward()  
                    self.optimizer.step()    
                    
                    sdr_metric = si_sdr_metric(to_linear(denormalize_db_spectr(x_stftspec_db_norm)),
                                               to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm)))
                    train_sdr_score += ((1./(n+1))*(sdr_metric-train_sdr_score))                                    
                                
                elif args.task == "spec2wav":
                    x_stft_mag = torch.abs(x_stft).float().unsqueeze(1)
                    
                    x_stft_hat_stack, x_stft_magreplaced = self.model(x_stft_mag)
                    x_stft_hat = r2_to_c(x_stft_magreplaced)
                    
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    
                    loss = self._compute_loss(r2_to_c(x_stft_hat_stack), x_stft)
                    train_loss += ((1./(n+1))*(loss-train_loss))
                    loss.backward()  
                    self.optimizer.step()

                    # Compute metrics
                    stoi_metric = self.stoi(x_wav, x_wav_hat) 
                    train_stoi_score += ((1./(n+1))*(stoi_metric-train_stoi_score))

                    pesq_metric = self.pesq(x_wav, x_wav_hat) 
                    train_pesq_score += ((1./(n+1))*(pesq_metric-train_pesq_score))
                    
                elif args.task == "mel2wav":
                    # TODO: not implemented yet 
                    # stack ConvPInv and DeGLI
                    pass
                
                else:
                    raise ValueError(f"task must be one of [melspec2spec, spec2wav, mel2wav], \
                        received {args.task}")
                    
                pbar.set_postfix_str(f'mse: {train_loss:.6f}, stoi: {train_stoi_score:.3f}, pesq: {train_pesq_score:.3f}')
                
                if n == 50:
                    break

            # Evaluate on the validation set
            val_stoi_score, val_loss = self.eval_model(model=self.model, 
                                                  test_dl=val_dl,
                                                  task=args.task)
            self.lr_sched.step(val_loss)
            # Update training state
            self._update_training_state(train_loss, train_stoi_score, val_loss, val_stoi_score)
            
            # Save the best model
            if self.training_state["patience_epochs"] == 0:
                torch.save(self.model.state_dict(), self.best_weights_path)
                    
            # Save checkpoint to resume training
            with open(self.training_state_path, "w") as fw:
                json.dump(self.training_state, fw, indent=4)    
            torch.save(self.model.state_dict(), self.ckpt_weights_path)
            torch.save(self.optimizer.state_dict(), self.ckpt_opt_path)
            torch.save(self.lr_sched.state_dict(), self.ckpt_sched_path)
            
            # Save plot of train history
            plot_train_hist(self.experiment_dir)            
            
            print(f'Training loss:     {train_loss.item():.6f} \t| Validation Loss:   {val_loss.item():.6f}')
            print(f'Training stoi:   {train_stoi_score.item():.4f} \t| Validation stoi: {val_stoi_score.item():.4f}')
            print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int(((time()-start_epoch))%60)} s')
            print('_____________________________')

        print('____________________________________________')
        print('Best epoch was Epoch ', self.training_state["best_epoch"])    
        print(f'Training loss:     {self.training_state["train_loss_hist"][self.training_state["best_epoch"]-1]:.6f} \t| Validation Loss:   {self.training_state["val_loss_hist"][self.training_state["best_epoch"]-1]}')
        print(f'Training metric:   {self.training_state["train_score_hist"][self.training_state["best_epoch"]-1]:.4f} \t| Validation metric: {self.training_state["best_val_score"]}')
        print('____________________________________________')

        return self.training_state
    
    def _create_noise(self, signal, max_nsr_db = 6):
    
        sdr_db = max_nsr_db * torch.rand((1)) - max_nsr_db
        sdr = torch.pow(10, sdr_db/10).to(self.hprms.device)

        signal_power = torch.mean(torch.abs(signal) ** 2)
        
        noise_power = signal_power / sdr
        noise = torch.sqrt(noise_power) * torch.randn(signal.shape, dtype=torch.complex64, device=self.hprms.device)
        
        return noise
    
    def _compute_loss(self, x_stft_hat_stack, x_stft):
        
        loss = 0.
        scale_factor = 0.
        for n in range(x_stft_hat_stack.shape[1]):
            x_stft_hat = x_stft_hat_stack[:,n]
            scale_factor += 1. / (self.hprms.n_degli_repetitions - n)
            loss += (self.loss(x_stft_hat, x_stft) / (self.hprms.n_degli_repetitions - n))
        loss /= scale_factor
        return loss
    
    def eval_model(self,
                   model: torch.nn.Module, 
                   test_dl: DataLoader,
                   task: str)->torch.Tensor:

        model.eval()

        test_score = 0.
        test_loss = 0.
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                x_stft = batch["stft"].to(model.device)
                
                if task == "melspec2spec":
                    x_stftspec_db_norm, x_melspec_db_norm = self._preprocess_mel2spec(x_stft)
                    x_stftspec_hat_db_norm = model(x_melspec_db_norm).squeeze()
                    
                    loss = mse(x_stftspec_db_norm, x_stftspec_hat_db_norm)
                    test_loss += ((1./(n+1))*(loss-test_loss))
                    
                    sdr_metric = si_sdr_metric(to_linear(denormalize_db_spectr(x_stftspec_db_norm)),
                                            to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm)))
                    test_score += ((1./(n+1))*(sdr_metric-test_score))  
                
                elif task == "spec2wav":
                    
                    x_stft_mag = torch.abs(x_stft).float().unsqueeze(1)
                      
                    x_stft_hat_stack, x_stft_magreplaced = self.model(x_stft_mag)
                    x_stft_hat = r2_to_c(x_stft_magreplaced)
                    
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    
                    loss = self._compute_loss(r2_to_c(x_stft_hat_stack), x_stft)
                    test_loss += ((1./(n+1))*(loss-test_loss))
                    
                    if x_wav.dim() == 1:
                        x_wav = x_wav.unsqueeze(0)
                        x_wav_hat = x_wav_hat.unsqueeze(0)
                        
                    stoi_metric = self.stoi(x_wav, x_wav_hat) 
                    test_score += ((1./(n+1))*(stoi_metric-test_score))

                
                pbar.set_postfix_str(f'mse: {test_loss:.6f}, si-sdr: {test_score:.3f}')  
                if n == 20:
                    break    
        return test_score, test_loss

    def _preprocess_mel2spec(self, x_stft):
        
        x_stftspec_db_norm = normalize_db_spectr(to_db(torch.abs(x_stft))).float()
        x_melspec_db_norm = torch.matmul(self.melfb, x_stftspec_db_norm).unsqueeze(1)
        
        return x_stftspec_db_norm, x_melspec_db_norm
    
    def _set_hparams(self, resume_training):
        
        if resume_training:
            self.hprms = config.load_config(self.config_path)
        else:
            self.hprms = config.create_hparams(args.model_name)
            config.save_config(self.config_path)
                
    def _set_paths(self, experiment_name, task):
        
        if task == "melspec2spec":
            results_dir = config.MELSPEC2SPEC_DIR 
        elif task == "spec2wav":
            results_dir = config.SPEC2WAV_DIR 
            
        self.experiment_dir = results_dir / experiment_name            
        self.experiment_weights_dir = config.WEIGHTS_DIR / experiment_name

        # json
        self.training_state_path = self.experiment_dir / "train_state.json"    
        self.config_path = self.experiment_dir / "config.json"
        # torch
        self.best_weights_path = self.experiment_weights_dir / 'best_weights'
        self.ckpt_weights_path = self.experiment_weights_dir / 'ckpt_weights'
        self.ckpt_opt_path = self.experiment_weights_dir / 'ckpt_opt'
        self.ckpt_sched_path = self.experiment_weights_dir / 'ckpt_sched'
        
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
                
        if not os.path.exists(self.experiment_weights_dir):
            os.mkdir(self.experiment_weights_dir) 
            
    def _update_training_state(self, train_loss, train_sdr_score, val_loss, val_stoi_score):
            
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            if isinstance(train_sdr_score, torch.Tensor):
                train_sdr_score = train_sdr_score.item()
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if isinstance(val_stoi_score, torch.Tensor):
                val_stoi_score = val_stoi_score.item()
                
            self.training_state["train_loss_hist"].append(train_loss)
            self.training_state["train_score_hist"].append(train_sdr_score)
            self.training_state["val_loss_hist"].append(val_loss)
            self.training_state["val_score_hist"].append(val_stoi_score)
            
            if val_stoi_score <= self.training_state["best_val_score"]:
                self.training_state["patience_epochs"] += 1
                print(f'\nBest epoch was Epoch {self.training_state["best_epoch"]}: Validation metric = {self.training_state["best_val_score"]}')
            else:
                self.training_state["patience_epochs"] = 0
                self.training_state["best_val_score"] = val_stoi_score
                self.training_state["best_val_loss"] = val_loss
                self.training_state["best_epoch"] = self.training_state["epochs"]
                print("\nMetric on validation set improved")


def main(args):
    
    trainer = Trainer(args)
    
    train_dl = build_dataloader(trainer.hprms, config.DATA_DIR, "train")
    val_dl = build_dataloader(trainer.hprms, config.DATA_DIR, "validation")
    
    _ = trainer.train(train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        choices=["unet", "convpinv", "degliblock", "degli"],
                        default='degli')
    parser.add_argument('--task',
                        type=str,
                        choices=["melspec2spec", "spec2wav"],
                        default='spec2wav')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test')
    parser.add_argument('--resume_training',
                        type=bool,
                        help="set to True if you want to restart training from a checkpoint",
                        default=False)
    
    args = parser.parse_args()
    main(args)