import argparse
import json
import os
from time import time

import librosa
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

import config
from dataset import build_dataloader
from losses import ComplexMSELoss, FrobeniusLoss, l2_regularization
from networks.build_model import build_model
from utils.audioutils import (compute_wav, denormalize_db_spectr,
                              normalize_db_spectr, to_db, to_linear)
from utils.plots import plot_train_hist, plot_train_hist_degli
from utils.utils import save_to_json

class Trainer:
    def __init__(self, args):
        
        self.model_name = args.model_name
        self.task = args.task
        self.data_degli_name = args.data_degli_name
        self._set_paths(args.task, args.experiment_name, args.data_degli_name)
        self._set_hparams(args.resume_training)
        self._set_loss(self.hprms.loss)
        
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = self.hprms.sr, 
                                                         n_fft = self.hprms.n_fft, 
                                                         n_mels = self.hprms.n_mels)).to(self.hprms.device)

        self.training_state = {"epochs": 0,
                                "patience_epochs": 0,  
                                "best_epoch": 0,
                                "best_epoch_scores": {"pesq": 0.,
                                                      "si-sdr": 0},
                                "train_hist": {},
                                "val_hist": {}}
        
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
        self.sisdr = ScaleInvariantSignalDistortionRatio().to(self.hprms.device)
        if args.resume_training:
            # Load model's weights, optimizer and scheduler from checkpoint
            with open(self.training_state_path, "r") as fp:
                self.training_state = json.load(fp)        
            self.model = build_model(self.hprms, args.model_name, self.experiment_weights_dir, best_weights = False)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)        
            self.optimizer.load_state_dict(torch.load(self.ckpt_opt_path)) 
            self.lr_sched = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=self.hprms.lr_patience)
            self.lr_sched.load_state_dict(torch.load(self.ckpt_sched_path))
        else:        
            # Initialize model, optimizer and lr scheduler
            self.model = build_model(self.hprms, args.model_name)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)
            self.lr_sched = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=self.hprms.lr_patience)
               
        if self.data_degli_name is not None:
            data_degli_hprms = config.load_config(self.data_degli_config_path)
            self.data_degli = build_model(data_degli_hprms, "degli", self.data_degli_weights_dir)
            
            
    def train(self, train_dl, val_dl):
        
        print('_____________________________')
        print('       Training start')
        print('_____________________________')
        
        while self.training_state["patience_epochs"] < self.hprms.patience and self.training_state["epochs"] < self.hprms.epochs:
            
            self.training_state["epochs"] += 1 
            print(f'\n§ Train Epoch: {self.training_state["epochs"]}\n')
            
            self.model.train()
            train_scores = {"loss": 0.,
                            "si-sdr": 0.,
                            "stoi": 0.,
                            "pesq": 0.}
            start_epoch = time()        
            pbar = tqdm(train_dl, desc=f'Epoch {self.training_state["epochs"]}', postfix='[]')
            
            for n, batch in enumerate(pbar):   
                
                self.optimizer.zero_grad()  
                
                if self.task == "melspec2spec":

                    x_stftspec_db_norm, x_melspec_db_norm = self._preprocess_mel2spec_batch(batch)
                    x_stftspec_hat_db_norm = self.model(x_melspec_db_norm).squeeze(1)
                    
                    loss = self.loss_fn(x_stftspec_db_norm, x_stftspec_hat_db_norm)
                    
                    if self.hprms.weights_decay is not None:
                        l2_reg = l2_regularization(self.model)
                        loss += self.hprms.weights_decay * l2_reg
                        
                    train_scores["loss"] += ((1./(n+1))*(loss-train_scores["loss"]))
                    loss.backward()  
                    self.optimizer.step()    
                    
                    sdr_metric = self.sisdr(to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm)),
                                            to_linear(denormalize_db_spectr(x_stftspec_db_norm))).detach()
                    
                    if (not torch.isnan(sdr_metric) and not torch.isinf(sdr_metric)):
                        train_scores["si-sdr"] += ((1./(n+1))*(sdr_metric-train_scores["si-sdr"]))
                                
                elif self.task == "spec2wav":
                    if self.data_degli_name is not None:
                        x_stft, x_stft_noise = self._preprocess_degli_deglidata_batch(batch)
                    else:
                        x_stft, x_stft_noise = self._preprocess_degli_awgndata_batch(batch)
                   
                    x_stft_hat = self.model(x_stft_noise, torch.abs(x_stft).float())
                    
                    loss = self.loss_fn(x_stft_hat, x_stft)
                    train_scores["loss"] += ((1./(n+1))*(loss-train_scores["loss"]))
                    loss.backward()  
                    self.optimizer.step()
                    
                    # Compute metrics
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()
                    
                    # STOI
                    stoi_metric = self.stoi(x_wav_hat, x_wav) 
                    train_scores["stoi"] += ((1./(n+1))*(stoi_metric-train_scores["stoi"]))
                    # PESQ (check if there is no utterance)
                    try:
                        pesq_metric = self.pesq(x_wav_hat, x_wav)
                    except:
                        pesq_metric = train_scores["pesq"]
                    train_scores["pesq"]  += ((1./(n+1))*(pesq_metric-train_scores["pesq"]))

                    
                elif self.task == "mel2wav":
                    # TODO: not implemented yet 
                    # stack ConvPInv and DeGLI
                    pass
                
                else:
                    raise ValueError(f"task must be one of [melspec2spec, spec2wav, mel2wav], \
                        received {self.task}")
                    
                scores_to_print = str({k: round(float(v), 4) for k, v in train_scores.items() if v != 0.})
                pbar.set_postfix_str(scores_to_print)
                
                if n == 100:
                    break

            # Evaluate on the validation set
            val_scores = self.eval_model(model = self.model, 
                                         test_dl = val_dl,
                                         task = self.task)
            if self.task == "mel2spec":
                self.lr_sched.step(val_scores["si-sdr"])
            elif self.task == "spec2wav":
                self.lr_sched.step(val_scores["pesq"])
            # Update and save training state
            self._update_training_state(train_scores, val_scores)
            self._save_training_state()
            # Save plot of train history
            if self.task == "spec2wav":
                plot_train_hist_degli(self.experiment_dir)
            else:
                plot_train_hist(self.experiment_dir)            

            print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int(((time()-start_epoch))%60)} s')
            print('_____________________________')

        print("____________________________________________")
        print("          Training completed")    
        print("____________________________________________")

        return self.training_state
    
    
    def _preprocess_mel2spec_batch(self, batch):
        
        x_stft = batch["stft"].to(self.hprms.device)
        x_stftspec = torch.abs(x_stft).float()
        x_melspec = torch.matmul(self.melfb, x_stftspec**2).unsqueeze(1)
        
        x_stftspec_db_norm = normalize_db_spectr(to_db(x_stftspec)).float()
        # x_stftspec_db_norm = normalize_db_spectr(to_db(torch.abs(x_stft))).float()
        x_melspec_db_norm = normalize_db_spectr(to_db(torch.sqrt(x_melspec))).float()
        # x_melspec_db_norm = torch.matmul(self.melfb, x_stftspec_db_norm).unsqueeze(1)

        return x_stftspec_db_norm, x_melspec_db_norm
    
    
    def _preprocess_degli_deglidata_batch(self, batch, max_degli_rep=10, min_degli_rep=1):
        
        self.data_degli.repetitions = torch.randint(min_degli_rep, max_degli_rep, size=(1,))
        with torch.no_grad():
            x_stft = batch["stft"].to(self.hprms.device)
            x_stft_noise = self._initialize_random_phase(x_stft)
            x_stft_noise = self.data_degli(x_stft_noise, torch.abs(x_stft).float())
        return x_stft, x_stft_noise
        
    
    def _preprocess_degli_awgndata_batch(self, batch):
        
        x_stft = batch["stft"].to(self.hprms.device)
        noise = self._create_noise(x_stft, 
                                   max_snr_db = self.hprms.max_snr_db,
                                   min_snr_db = self.hprms.min_snr_db)
        
        x_stft_noise = x_stft + noise
        
        return x_stft, x_stft_noise
    
    
    def _initialize_random_phase(self, x_stft):
        
        x_stft_mag = torch.abs(x_stft)
        phase = torch.zeros_like(x_stft_mag)
        x_stft = x_stft_mag * torch.exp(1j * phase)
        return x_stft
    
    
    def _create_noise(self, signal, max_snr_db = 12, min_snr_db = -6):
    
        sdr_db = (max_snr_db - min_snr_db) * torch.rand((1)) + min_snr_db
        sdr = torch.pow(10, sdr_db/10).to(signal.device)

        signal_power = torch.mean(torch.abs(signal) ** 2)
        
        noise_power = signal_power / (sdr + 1e-12)
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        
        return noise
    
    
    def eval_model(self,
                   model: torch.nn.Module, 
                   test_dl: DataLoader,
                   task: str)->torch.Tensor:

        model.eval()

        test_scores = {"loss": 0.,
                        "si-sdr": 0.,
                        "stoi": 0.,
                        "pesq": 0.}        
        pbar = tqdm(test_dl, desc=f'Evaluation', postfix='[]')
        with torch.no_grad():
            for n, batch in enumerate(pbar):   
                
                if task == "melspec2spec":
                    x_stftspec_db_norm, x_melspec_db_norm = self._preprocess_mel2spec_batch(batch)
                    x_stftspec_hat_db_norm = model(x_melspec_db_norm).squeeze(1)
                    
                    loss = self.loss_fn(x_stftspec_db_norm, x_stftspec_hat_db_norm)
                    if self.hprms.weights_decay is not None:
                        l2_reg = l2_regularization(self.model)
                        loss += self.hprms.weights_decay * l2_reg
                        
                    test_scores["loss"] += ((1./(n+1))*(loss-test_scores["loss"]))
                    
                    sdr_metric = self.sisdr(to_linear(denormalize_db_spectr(x_stftspec_hat_db_norm)),
                                            to_linear(denormalize_db_spectr(x_stftspec_db_norm)))
                    test_scores["si-sdr"] += ((1./(n+1))*(sdr_metric-test_scores["si-sdr"]))  
                
                elif task == "spec2wav":
                    x_stft = batch["stft"].to(model.device)
                    x_stft_noise = self._initialize_random_phase(x_stft)
                    x_stft_hat = self.model(x_stft_noise, torch.abs(x_stft).float())
                    
                    x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                    x_wav_hat = compute_wav(x_stft_hat, n_fft=self.hprms.n_fft).squeeze().detach()                
                    if x_wav.dim() == 1:
                        x_wav = x_wav.unsqueeze(0)
                        x_wav_hat = x_wav_hat.unsqueeze(0)
                    
                    # Compute metrics
                    stoi_metric = self.stoi(x_wav_hat, x_wav) 
                    test_scores["stoi"] += ((1./(n+1))*(stoi_metric-test_scores["stoi"]))
                        
                    try:
                        pesq_metric = self.pesq(x_wav_hat, x_wav) 
                    except:
                        pesq_metric = test_scores["pesq"]
                    test_scores["pesq"] += ((1./(n+1))*(pesq_metric-test_scores["pesq"]))

                elif task == "melspec2wav":
                    # TODO
                    pass
                
                scores_to_print = str({k: round(float(v), 4) for k, v in test_scores.items() if v != 0.})
                pbar.set_postfix_str(scores_to_print)
                
                if n == 50:
                    break  
                 
        return test_scores

    
    def _set_hparams(self, resume_training):

        if resume_training:
            self.hprms = config.load_config(self.config_path)
        else:
            self.hprms = config.create_hparams(self.model_name)
            config.save_config(self.hprms, self.config_path)
    
    def _set_loss(self, loss: str):
   
        if loss == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss == "complexmse":
            self.loss_fn = ComplexMSELoss()
        elif loss == "frobenius":
            self.loss_fn = FrobeniusLoss()
                      
    def _set_paths(self, 
                   task, 
                   experiment_name, 
                   data_degli_name: None):
   
        if task == "melspec2spec":
            results_dir = config.MELSPEC2SPEC_DIR 
        elif task == "spec2wav":
            results_dir = config.SPEC2WAV_DIR 
            if data_degli_name is not None:
                self.data_degli_weights_dir = config.WEIGHTS_DIR / data_degli_name
                self.data_degli_config_path = results_dir / data_degli_name / "config.json"
            
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
            
    def _update_training_state(self, train_scores, val_scores):

        train_scores = {k: round(float(v), 4) for k, v in train_scores.items() if v != 0.}
        val_scores = {k: round(float(v), 4) for k, v in val_scores.items() if v != 0.}
        
        for key, value in train_scores.items():
            if key not in self.training_state["train_hist"]:
                self.training_state["train_hist"][key] = []
            self.training_state["train_hist"][key].append(value)
        
        for key, value in val_scores.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.training_state["val_hist"]:
                self.training_state["val_hist"][key] = []
            self.training_state["val_hist"][key].append(value)
        
        if self.task == "spec2wav":
            metr = "pesq"
        elif self.task == "melspec2spec":
            metr = "si-sdr"
            
        if val_scores[metr] <= self.training_state["best_epoch_scores"][metr]:
            self.training_state["patience_epochs"] += 1
            print(f'\nBest epoch was Epoch {self.training_state["best_epoch"]}: Validation metric = {self.training_state["best_epoch_scores"][metr]}')
        else:
            self.training_state["patience_epochs"] = 0
            self.training_state["best_epoch"] = self.training_state["epochs"]
            self.training_state["best_epoch_scores"] = {k: v[self.training_state["best_epoch"]-1] for k,v in self.training_state["val_hist"].items()}
            print("Metric on validation set improved")


    def _save_training_state(self):
        # Save the best model
        if self.training_state["patience_epochs"] == 0:
            torch.save(self.model.state_dict(), self.best_weights_path)
                
        # Save checkpoint to resume training
        save_to_json(self.training_state, self.training_state_path)  
        torch.save(self.model.state_dict(), self.ckpt_weights_path)
        torch.save(self.optimizer.state_dict(), self.ckpt_opt_path)
        torch.save(self.lr_sched.state_dict(), self.ckpt_sched_path)
        
        
def main(args):
    
    trainer = Trainer(args)
    
    train_dl = build_dataloader(trainer.hprms, config.DATA_DIR, "train")
    val_dl = build_dataloader(trainer.hprms, config.DATA_DIR, "validation")
    
    _ = trainer.train(train_dl, val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        choices=["unet", "pinvconv", "pinvunet", "degli"],
                        default='pinvconv')
    
    parser.add_argument('--task',
                        type=str,
                        choices=["melspec2spec", "spec2wav"],
                        default='melspec2spec')
    
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test')
    
    parser.add_argument('--resume_training',
                        action='store_true',
                        help="use this flag if you want to restart training from a checkpoint")

    parser.add_argument('--data_degli_name',
                        type=str,
                        default=None)
    
    args = parser.parse_args()
    main(args)