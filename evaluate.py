from pathlib import Path
import json
import argparse
from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import librosa
import numpy as np
import os 

from model import build_model, ConvPInv
from metrics import si_snr_metric, mse
from dataset import build_dataloaders
from audioutils import to_linear, denormalize_db_spectr
import config



def eval_librosa(hparams: Namespace,
                 dataloader: DataLoader):

    test_score = 0.
    test_loss = 0.
    melfb = librosa.filters.mel(sr = hparams.sr, 
                                n_fft = hparams.n_fft, 
                                n_mels = hparams.n_mels)  
      
    for n, batch in enumerate(tqdm(dataloader)):
        stftspec_db_norm = batch["spectr"].squeeze()
        melspec_db_norm = np.dot(melfb, stftspec_db_norm.numpy())
        stftspec_hat_db_norm = librosa.feature.inverse.mel_to_stft(melspec_db_norm, 
                                                                   sr = hparams.sr,
                                                                   n_fft = hparams.n_fft)
        loss = mse(torch.as_tensor(stftspec_hat_db_norm), stftspec_db_norm)
        test_loss += ((1./(n+1))*(loss-test_loss))
                        
        score = si_snr_metric(to_linear(denormalize_db_spectr(torch.as_tensor(stftspec_hat_db_norm))), 
                                to_linear(denormalize_db_spectr(stftspec_db_norm)))
        test_score += ((1./(n+1))*(score-test_score))

    return test_loss, test_score
        
        
def eval_model(model: torch.nn.Module, 
               dataloader: DataLoader)->torch.Tensor:

    model.eval()

    val_score = 0.
    val_loss = 0.
    
    with torch.no_grad():
        for n, batch in enumerate(tqdm(dataloader)):
            stftspec_db_norm = batch["spectr"].float().to(config.DEVICE)
            melspec_db_norm = torch.matmul(model.pinvblock.melfb.float(), stftspec_db_norm)
            melspec_db_norm = melspec_db_norm.unsqueeze(1)
            
            stftspec_hat_db_norm = model(melspec_db_norm)
            
            loss = mse(stftspec_hat_db_norm, stftspec_db_norm)
            val_loss += ((1./(n+1))*(loss-val_loss))
                         
            score = si_snr_metric(to_linear(denormalize_db_spectr(stftspec_hat_db_norm)), 
                                    to_linear(denormalize_db_spectr(stftspec_db_norm)))
            val_score += ((1./(n+1))*(score-val_score))
            
    return val_score, val_loss
   

def main(args):
    
    config_path = config.MELSPEC2SPEC_DIR / args.experiment_name / "config.json"
    hparams = config.load_config(config_path)
    
    _, _, test_dl = build_dataloaders(config.DATA_DIR, hparams)
    test_metrics_path = config.MELSPEC2SPEC_DIR / args.experiment_name / 'test_metrics.json'
    if args.model_name == "librosa":
        if not os.path.exists(config.MELSPEC2SPEC_DIR / args.model_name):
            os.mkdir(config.MELSPEC2SPEC_DIR / args.model_name)       
        test_loss, test_score = eval_librosa(hparams, test_dl)
    else:
        model = build_model(hparams, args.model_name, args.experiment_name,  args.best_weights)
        test_score, test_loss = eval_model(model, test_dl)    
        
    test_metrics = {"mse": float(test_loss),
                    "si-snr": float(test_score)}
        
    with open(test_metrics_path, "w") as fp:
        json.dump(test_metrics, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        choices = ["unet", "librosa", "convpinv", "pinv"],
                        help = "models: unet, librosa (evaluates librosa.feature.inverse.mel_to_stft())," 
                        "convpinv (simple CNN + pseudoinverse melfb), pinv (pseudoinverse melfb baseline)",
                        type=str,
                        default = 'convpinv')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='convpinvL2K31EX200')
    parser.add_argument('--best_weights',
                        type=bool,
                        default=True)
    args = parser.parse_args()
    main(args)