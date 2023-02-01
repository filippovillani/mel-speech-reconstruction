import torch
import os
import json
import argparse
from time import time
from tqdm import tqdm

from model import build_model
from dataset import build_dataloader
from evaluate import eval_model
from metrics import si_snr_metric, mse
from plots import plot_train_hist
from audioutils import to_linear, denormalize_db_spectr
import config


def train_model(args):
    
    experiment_dir = config.MELSPEC2SPEC_DIR / args.experiment_name
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_weights_dir = config.WEIGHTS_DIR / args.experiment_name
    if not os.path.exists(experiment_weights_dir):
        os.mkdir(experiment_weights_dir)
        
    # json
    training_state_path = experiment_dir / "train_state.json"    
    config_path = experiment_dir / "config.json"
    # torch
    best_weights_path = experiment_weights_dir / 'best_weights'
    ckpt_weights_path = experiment_weights_dir / 'ckpt_weights'
    ckpt_opt_path = experiment_weights_dir / 'ckpt_opt'
    
    if args.experiment_weights_dir is not None:
        hparams = config.load_config(config_path)
        
        ckpt_opt_toload_path = config.WEIGHTS_DIR / args.experiment_weights_dir / 'ckpt_opt'
        
        # Load training state
        with open(training_state_path, "r") as fp:
            training_state = json.load(fp)
        
        # Load model's weights and optimizer from checkpoint
        model = build_model(hparams, args.model_name, args.experiment_weights_dir, best_weights = False)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr)        
        optimizer.load_state_dict(torch.load(ckpt_opt_toload_path))        
        
         
    else:
        hparams = config.create_hparams()
        config.save_config(config_path)
        
        model = build_model(hparams, args.model_name)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams.lr)

            
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
    train_dl = build_dataloader(hparams, config.DATA_DIR, "train") 
    val_dl = build_dataloader(hparams, config.DATA_DIR, "validation") 
    print('_____________________________')
    print('       Training start')
    print('_____________________________')
    while training_state["patience_epochs"] < hparams.patience and training_state["epochs"] < hparams.epochs:
        training_state["epochs"] += 1 
        print(f'\n§ Train epoch: {training_state["epochs"]}\n')
        
        model.train()
        start_epoch = time()        
        train_scores = {"loss": 0.,
                        "si-sdr": 0.}
        pbar = tqdm(train_dl, desc=f'Epoch {training_state["epochs"]}', postfix='[]')
        for n, batch in enumerate(pbar):   
            optimizer.zero_grad()  
            stftspec_db_norm = batch["spectrogram"].float().to(hparams.device)
            melspec_db_norm = torch.matmul(model.pinvblock.melfb, stftspec_db_norm).unsqueeze(1)
            
            stftspec_hat_db_norm = model(melspec_db_norm).squeeze()
            
            loss = mse(stftspec_db_norm, stftspec_hat_db_norm)
            train_scores["loss"] += ((1./(n+1))*(loss-train_scores["loss"]))
            loss.backward()  
            optimizer.step()

            snr_metric = si_snr_metric(to_linear(denormalize_db_spectr(stftspec_db_norm)),
                                       to_linear(denormalize_db_spectr(stftspec_hat_db_norm)))
            train_scores["si-sdr"] += ((1./(n+1))*(snr_metric-train_scores["si-sdr"]))

            # if n == 100:
            #     break
        
            scores_to_print = str({k: round(float(v), 4) for k, v in train_scores.items()})
            pbar.set_postfix_str(scores_to_print)
            
        training_state["train_loss_hist"].append(train_scores["loss"].item())
        training_state["train_score_hist"].append(train_scores["si-sdr"].item())
        print(f'Training loss:     {training_state["train_loss_hist"][-1]:.4f}')
        print(f'Training SI-SNR:   {training_state["train_score_hist"][-1]:.4f} dB\n')
        
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
            print(f'\nBest epoch was Epoch {training_state["best_epoch"]}: Validation SI-SNR = {training_state["best_val_score"]} dB')
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
            json.dump(training_state, fw, indent=4)
            
        torch.save(model.state_dict(), ckpt_weights_path)
        torch.save(optimizer.state_dict(), ckpt_opt_path)

        print(f'Epoch time: {int(((time()-start_epoch))//60)} min {int((((time()-start_epoch))%60)*60/100)} s')
        print('_____________________________')

    print('Best epoch was Epoch ', training_state["best_epoch"])    
    print('val MSE Loss    :  \t', training_state["val_loss_hist"][training_state["best_epoch"]-1])
    print('val SI-SNR Score: \t', training_state["best_val_score"])
    print('____________________________________________')

def main(args):
    
    train_model(args)
    plot_train_hist(args.experiment_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        type=str,
                        default='test')
    parser.add_argument('--experiment_weights_dir',
                        type=str,
                        help="directory containing the the model's checkpoint weights",
                        default=None)
    parser.add_argument('--model_name',
                        type=str,
                        choices=["unet", "convpinv"],
                        default='convpinv')
    
    args = parser.parse_args()
    main(args)