import torch
from torch.utils.data import DataLoader
import os
import json
import argparse
from time import time
from tqdm import tqdm

from model import MelSpect2Spec
from dataset import build_dataloaders
from metrics import si_nsr_loss, si_ssnr_metric, mse
from plots import plot_train_hist
from audioutils import db_to_amplitude
import config

def eval_model(model: torch.nn.Module, 
               dataloader: DataLoader)->torch.Tensor:

    model.eval()

    score = 0.
    loss = 0.
    
    with torch.no_grad():
        for n, batch in enumerate(tqdm(dataloader)):
            melspec = batch["melspectr"].to(config.DEVICE)

            melspec_hat = model(melspec.float())
            
            nsr_loss = mse(melspec_hat, melspec)
            loss += ((1./(n+1))*(nsr_loss-loss))
                         
            snr_metric = si_ssnr_metric(db_to_amplitude(melspec_hat), db_to_amplitude(melspec))
            score += ((1./(n+1))*(snr_metric-score))

    return score, loss

def train_model(args, hparams):
    
    experiment_dir = config.MELSPEC2SPEC_DIR / args.experiment_name
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
        
    model = MelSpect2Spec(hparams).float().to(config.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=hparams.lr)

    if args.weights_dir is not None:
        best_weights_path = config.WEIGHTS_DIR / args.experiment_name / 'best_weights'
        ckpt_weights_path = config.WEIGHTS_DIR / args.experiment_name / 'ckpt_weights'
        ckpt_opt_path = config.WEIGHTS_DIR / args.experiment_name / 'ckpt_opt'
        
        training_state_path = config.MELSPEC2SPEC_DIR / args.weights_dir / "train_state.json"
        ckpt_weights_toload_path = config.WEIGHTS_DIR / args.weights_dir / 'ckpt_weights'
        ckpt_opt_toload_path = config.WEIGHTS_DIR / args.weights_dir / 'ckpt_opt'
        
        model.load_state_dict(torch.load(ckpt_weights_toload_path))
        optimizer.load_state_dict(torch.load(ckpt_opt_toload_path))        
        
        with open(training_state_path, "r") as fr:
            training_state = json.load(fr)
         
    else:
        training_state_path = experiment_dir / "train_state.json"
        weights_dir = config.WEIGHTS_DIR / args.experiment_name
        best_weights_path = weights_dir / 'best_weights'
        ckpt_weights_path = weights_dir / 'ckpt_weights'
        ckpt_opt_path = weights_dir / 'ckpt_opt'
        
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
            
        training_state = {"epochs": 0,
                          "patience_epochs": 0,  
                          "best_val_loss": 9999,
                          "best_val_score": 0,
                          "best_epoch": 0,
                          "train_loss_hist": [],
                          "train_score_hist": [],
                          "val_loss_hist": [],
                          "val_score_hist": []}

    # Build training and validation 
    train_dl, val_dl = build_dataloaders(config.DATA_DIR, hparams) 

    print('_____________________________')
    print('       Training start')
    print('_____________________________')
    while training_state["patience_epochs"] < hparams.patience and training_state["epochs"] <= hparams.epochs:
        training_state["epochs"] += 1 
        print(f'\n§ Train epoch: {training_state["epochs"]}\n')
        
        model.train()
        train_loss = 0.
        train_score = 0.
        start_epoch = time()        
   
        for n, batch in enumerate(tqdm(train_dl, desc=f'Epoch {training_state["epochs"]}')):   
            optimizer.zero_grad()  
            melspec = batch["melspectr"].float().to(config.DEVICE)

            melspec_hat = model(melspec)
            
            loss = mse(melspec_hat, melspec)
            train_loss += ((1./(n+1))*(loss-train_loss))
            loss.backward()  
            optimizer.step()

            snr_metric = si_ssnr_metric(db_to_amplitude(melspec_hat), db_to_amplitude(melspec))
            train_score += ((1./(n+1))*(snr_metric-train_score))
            
        training_state["train_loss_hist"].append(train_loss.item())
        training_state["train_score_hist"].append(train_score.item())
        print(f'Training loss:     {training_state["train_loss_hist"][-1]:.4f}')
        print(f'Training SI-SNR:   {training_state["train_loss_hist"][-1]:.4f} dB\n')
        
        # Evaluate on the validation set
        print(f'Evaluating the model on validation set...')
        val_score, val_loss = eval_model(model=model, 
                                         dataloader=val_dl)
        
        training_state["val_loss_hist"].append(val_loss.item())
        training_state["val_score_hist"].append(val_score.item())
        
        print(f'Validation Loss:   {val_loss:.4f}')
        print(f'Validation SI-SNR: {val_score:.4f} dB\n')
        
        if val_score <= training_state["best_val_score"]:
            training_state["patience_epochs"] += 1
            print(f'\nBest epoch was Epoch {training_state["best_epoch"]}: val_score={training_state["best_val_score"]} dB')
        else:
            training_state["patience_epochs"] = 0
            training_state["best_val_score"] = val_score.item()
            training_state["best_val_loss"] = val_loss.item()
            training_state["best_epoch"] = training_state["epochs"]
            print("\nSI-SNR on validation set improved")
            # Save the best model
            torch.save(model.state_dict(), best_weights_path)
                   
        # Save checkpoint to resume training
        with open(training_state_path, "w") as fw:
            json.dump(training_state, fw)
            
        torch.save(model.state_dict(), ckpt_weights_path)
        torch.save(optimizer.state_dict(), ckpt_opt_path)

        print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
        print('_____________________________')

    print('Best epoch was Epoch ', training_state["best_epoch"])    
    print('val SI-NSR Loss:  \t', training_state["val_loss_hist"][training_state["best_epoch"]-1])
    print('val SI-SNR Score: \t', training_state["best_val_score"])
    print('____________________________________________')

def main(args):
    train_model(args, config.create_hparams())
    plot_train_hist(args.experiment_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        type=str,
                        default='prova04')
    parser.add_argument('--weights_dir',
                        type=str,
                        default=None)
    
    args = parser.parse_args()
    main(args)