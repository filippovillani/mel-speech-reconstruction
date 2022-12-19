from pathlib import Path
import json
import argparse
from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from metrics import si_snr_metric, mse
from dataset import build_dataloaders
from audioutils import to_linear, denormalize_db_spectr
import config

def eval_model(model: torch.nn.Module, 
               dataloader: DataLoader)->torch.Tensor:

    model.eval()

    val_score = 0.
    val_loss = 0.
    
    with torch.no_grad():
        for n, batch in enumerate(tqdm(dataloader)):
            stftspec_db_norm = batch["spectr"].float().to(config.DEVICE)
            melspec_db_norm = torch.matmul(model.pinvblock.melfb, stftspec_db_norm)
            melspec_db_norm = melspec_db_norm.unsqueeze(1)
            
            stftspec_hat_db_norm = model(melspec_db_norm)
            
            loss = mse(stftspec_hat_db_norm, stftspec_db_norm)
            val_loss += ((1./(n+1))*(loss-val_loss))
                         
            score = si_snr_metric(to_linear(denormalize_db_spectr(stftspec_hat_db_norm)), 
                                    to_linear(denormalize_db_spectr(stftspec_db_norm)))
            val_score += ((1./(n+1))*(score-val_score))
            
    return val_score, val_loss

def build_model(weights_dir: Path, 
                hparams: Namespace,
                best_weights: bool = True):
     
    weights_path = 'best_weights' if best_weights else 'ckpt_weights'
    weights_path = config.WEIGHTS_DIR / weights_dir / weights_path
    
    model = UNet(hparams).float().to(config.DEVICE)
    model.load_state_dict(torch.load(weights_path))
    
    return model    

def main(args):
    
    hparams = config.create_hparams()
    model = build_model(args.weights_dir, hparams, args.best_weights)
    _, _, test_dl = build_dataloaders(config.DATA_DIR, hparams)
    test_score, test_loss = eval_model(model, test_dl)
    test_metrics = {"mse": float(test_loss),
                    "si-snr": float(test_score)}
    test_metrics_path = config.MELSPEC2SPEC_DIR / args.weights_dir / 'test_metrics.json'
        
    with open(test_metrics_path, "w") as fp:
        json.dump(test_metrics, fp)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir',
                        type=str,
                        default='unet4_64')
    parser.add_argument('--best_weights',
                        type=bool,
                        default=True)
    args = parser.parse_args()
    main(args)