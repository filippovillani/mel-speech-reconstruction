import torch
import os
import json
import argparse
from time import time
from tqdm import tqdm

from networks.build_model import build_model
from dataset import build_dataloader
from evaluate import eval_model
from metrics import si_snr_metric, mse
from utils.plots import plot_train_hist
from utils.audioutils import to_linear, denormalize_db_spectr
import config

class Trainer:
    def __init__(self, args):
        
        self._set_paths(args.experiment_name)
        self._set_hparams(args.resume_training)

        if args.resume_training:
            # Load training state
            with open(self.training_state_path, "r") as fp:
                self.training_state = json.load(fp)
        
            # Load model's weights and optimizer from checkpoint  
            self.model = build_model(self.hprms, args.model_name, self.experiment_weights_dir, best_weights = False)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)        
            self.optimizer.load_state_dict(torch.load(self.ckpt_opt_path)) 
            
        else:        
            self.model = build_model(self.hprms, args.model_name)
            self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.hprms.lr)

                
            self.training_state = {"epochs": 0,
                                   "patience_epochs": 0,  
                                   "best_epoch": 0,
                                   "best_val_loss": 9999,
                                   "best_val_score": 0,
                                   "train_loss_hist": [],
                                   "train_score_hist": [],
                                   "val_loss_hist": [],
                                   "val_score_hist": []} 
        
    def _set_hparams(self, resume_training):
        
        if resume_training:
            self.hprms = config.load_config(self.config_path)
        else:
            self.hprms = config.create_hparams()
            config.save_config(self.config_path)
                
    def _set_paths(self, experiment_name):
        
        self.experiment_dir = config.MELSPEC2SPEC_DIR / experiment_name            
        self.experiment_weights_dir = config.WEIGHTS_DIR / experiment_name

        # json
        self.training_state_path = self.experiment_dir / "train_state.json"    
        self.config_path = self.experiment_dir / "config.json"
        # torch
        self.best_weights_path = self.experiment_weights_dir / 'best_weights'
        self.ckpt_weights_path = self.experiment_weights_dir / 'ckpt_weights'
        self.ckpt_opt_path = self.experiment_weights_dir / 'ckpt_opt'
        
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
                
        if not os.path.exists(self.experiment_weights_dir):
            os.mkdir(self.experiment_weights_dir)        
    
    def train(self, train_dl, val_dl):
        
        print('_____________________________')
        print('       Training start')
        print('_____________________________')
        while self.training_state["patience_epochs"] < self.hprms.patience and self.training_state["epochs"] < self.hprms.epochs:
            
            self.training_state["epochs"] += 1 
            print(f'\n§ Train Epoch: {self.training_state["epochs"]}\n')
            
            self.model.train()
            train_loss = 0.
            train_score = 0.
            start_epoch = time()        
            pbar = tqdm(train_dl, desc=f'Epoch {self.training_state["epochs"]}', postfix='[]')
            for n, batch in enumerate(pbar):   
                self.optimizer.zero_grad()  
                stftspec_db_norm = batch["spectrogram"].float().to(self.hprms.device)
                melspec_db_norm = torch.matmul(self.model.pinvblock.melfb, stftspec_db_norm).unsqueeze(1)
                
                stftspec_hat_db_norm = self.model(melspec_db_norm).squeeze()
                
                loss = mse(stftspec_db_norm, stftspec_hat_db_norm)
                train_loss += ((1./(n+1))*(loss-train_loss))
                loss.backward()  
                self.optimizer.step()

                snr_metric = si_snr_metric(to_linear(denormalize_db_spectr(stftspec_db_norm)),
                                        to_linear(denormalize_db_spectr(stftspec_hat_db_norm)))
                train_score += ((1./(n+1))*(snr_metric-train_score))
                
                pbar.set_postfix_str(f'mse: {train_loss:.6f}, si-snr: {train_score:.3f}')
                
                if n == 10:
                    break

            # Evaluate on the validation set
            val_score, val_loss = eval_model(model=self.model, 
                                             dataloader=val_dl)
            
            print(f'Training loss:     {train_loss.item():.6f} \t| Validation Loss:   {val_loss.item():.6f}')
            print(f'Training SI-SNR:   {train_score.item():.4f} dB \t| Validation SI-SNR: {val_score.item():.4f} dB')
            
            
            # Update training state and plot history
            self._update_training_state(train_loss, train_score, val_loss, val_score)
            plot_train_hist(self.experiment_dir)
            
            # Save the best model
            if self.training_state["patience_epochs"] == 0:
                torch.save(self.model.state_dict(), self.best_weights_path)
                    
            # Save checkpoint to resume training
            with open(self.training_state_path, "w") as fw:
                json.dump(self.training_state, fw, indent=4)    
            torch.save(self.model.state_dict(), self.ckpt_weights_path)
            torch.save(self.optimizer.state_dict(), self.ckpt_opt_path)
            
            
            print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
            print('_____________________________')

        print('____________________________________________')
        print('Best epoch was Epoch ', self.training_state["best_epoch"])    
        print(f'Training loss:     {self.training_state["train_loss_hist"][self.training_state["best_epoch"]-1]:.6f} \t| Validation Loss:   {self.training_state["val_loss_hist"][self.training_state["best_epoch"]-1]}')
        print(f'Training SI-SNR:   {self.training_state["train_score_hist"][self.training_state["best_epoch"]-1]:.4f} dB \t| Validation SI-SNR: {self.training_state["best_val_score"]} dB')
        print('____________________________________________')

        return self.training_state

    def _update_training_state(self, train_loss, train_score, val_loss, val_score):
            
            self.training_state["train_loss_hist"].append(train_loss.item())
            self.training_state["train_score_hist"].append(train_score.item())
            self.training_state["val_loss_hist"].append(val_loss.item())
            self.training_state["val_score_hist"].append(val_score.item())
            
            if val_score <= self.training_state["best_val_score"]:
                self.training_state["patience_epochs"] += 1
                print(f'\nBest epoch was Epoch {self.training_state["best_epoch"]}: Validation SI-SNR = {self.training_state["best_val_score"]} dB')
            else:
                self.training_state["patience_epochs"] = 0
                self.training_state["best_val_score"] = val_score.item()
                self.training_state["best_val_loss"] = val_loss.item()
                self.training_state["best_epoch"] = self.training_state["epochs"]
                print("\nSI-SNR on validation set improved")


def main(args):
    
    trainer = Trainer(args)
    
    train_dl = build_dataloader(trainer.hprms, config.DATA_DIR, args.task , "train")
    val_dl = build_dataloader(trainer.hprms, config.DATA_DIR, args.task, "validation")
    
    training_state = trainer.train(train_dl, val_dl)
    print(training_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        choices=["unet", "convpinv", "degli"],
                        default='convpinv')
    parser.add_argument('--task',
                        type=str,
                        choices=["mel2stft", "stft2wav"],
                        default='mel2stft')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test')
    parser.add_argument('--resume_training',
                        type=bool,
                        help="set to True if you want to restart training from a checkpoint",
                        default=False)
    
    args = parser.parse_args()
    main(args)