import argparse
import json
import os
from argparse import Namespace

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import build_dataloader
from metrics import mse, si_sdr_metric
from networks.build_model import build_model
from train import Trainer
from utils.audioutils import denormalize_db_spectr, to_linear


def eval_librosa(hparams: Namespace,
                 dataloader: DataLoader):

    test_score = 0.
    test_loss = 0.
    melfb = librosa.filters.mel(sr = hparams.sr, 
                                n_fft = hparams.n_fft, 
                                n_mels = hparams.n_mels)  
      
    for n, batch in enumerate(tqdm(dataloader)):
        stftspec_db_norm = batch["stft"].squeeze()
        melspec_db_norm = np.dot(melfb, stftspec_db_norm.numpy())
        stftspec_hat_db_norm = librosa.feature.inverse.mel_to_stft(melspec_db_norm, 
                                                                   sr = hparams.sr,
                                                                   n_fft = hparams.n_fft)
        loss = mse(torch.as_tensor(stftspec_hat_db_norm), stftspec_db_norm)
        test_loss += ((1./(n+1))*(loss-test_loss))
                        
        score = si_sdr_metric(to_linear(denormalize_db_spectr(torch.as_tensor(stftspec_hat_db_norm))), 
                                to_linear(denormalize_db_spectr(stftspec_db_norm)))
        test_score += ((1./(n+1))*(score-test_score))

    return test_loss, test_score

    

def main(args):
    # TODO: divide these sections into functions
    
    # Directories
    if args.task == "melspec2spec":
        results_dir = config.MELSPEC2SPEC_DIR
    elif args.task == "spec2wav":
        results_dir = config.SPEC2WAV_DIR
    else:
        raise ValueError(f"task must be one of ['melspec2spec', 'spec2wav'], received {args.task}")
    
    config_path = results_dir / args.experiment_name / "config.json"
    weights_dir = config.WEIGHTS_DIR / args.experiment_name
    test_metrics_path = results_dir / args.experiment_name / 'test_metrics.json' # TODO
 
    if args.model_name != 'pinv':
        hparams = config.load_config(config_path)
    else:
        hparams = config.create_hparams()
    
    # Test DataLoader
    test_dl = build_dataloader(hparams, config.DATA_DIR, "test")

    # Model evaluation
    if args.model_name == "librosa":
        if not os.path.exists(config.MELSPEC2SPEC_DIR / args.model_name):
            os.mkdir(config.MELSPEC2SPEC_DIR / args.model_name)       
        test_loss, test_score = eval_librosa(hparams, test_dl)
    
    elif args.model_name in ["convpinv", "unet"]:
        trainer = Trainer(args)
        model = build_model(hparams, args.model_name, weights_dir, args.best_weights)
        test_scores = trainer.eval_model(model, test_dl, args.task)
        # test_metrics = {"mse": float(test_scores["loss"]),
        #                 "si-sdr": float(test_scores["sdr"])}
        with open(test_metrics_path, "w") as fp:
            json.dump(test_scores, fp, indent=4)    
    # TODO:
    elif args.model_name in ["degli"]:
        trainer = Trainer(args)
        model = build_model(hparams, args.model_name, weights_dir, args.best_weights)
        test_scores = trainer.eval_model(model, test_dl, args.task)   
        # test_metrics = {"mse": float(test_scores["loss"]),
        #                 "si-sdr": float(test_scores["sdr"])}
        with open(test_metrics_path, "w") as fp:
            json.dump(test_scores, fp, indent=4) 
    else: 
        raise ValueError(f'model_name must be one of ["unet", "librosa", "convpinv", "pinv", "degli"], \
                         received: {args.model_name}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        choices = ["unet", "librosa", "convpinv", "pinv", "degli"],
                        help = "models: unet, librosa (evaluates librosa.feature.inverse.mel_to_stft())," 
                        "convpinv (simple CNN + pseudoinverse melfb), pinv (pseudoinverse melfb baseline)",
                        type=str,
                        default = 'degli')
    parser.add_argument('--task',
                        type=str,
                        choices=["melspec2spec", "spec2wav"],
                        default='spec2wav')
    parser.add_argument('--experiment_name',
                        type=str,
                        default='degli_B1_K33_C32_N5')
    parser.add_argument('--best_weights',
                        type=bool,
                        default=True)
    parser.add_argument('--resume_training',
                        action='store_true',
                        help="use this flag if you want to restart training from a checkpoint")
    args = parser.parse_args()
    main(args)