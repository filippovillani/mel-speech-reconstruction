import argparse
import json

import librosa
import soundfile as sf
import torch

import config
from griffinlim import fast_griffin_lim
from metrics import mse, si_snr_metric
from networks.build_model import build_model
from utils.audioutils import (denormalize_db_spectr, normalize_db_spectr,
                              open_audio, to_db, to_linear)
from utils.plots import plot_prediction


def predict(args):
    
    experiment_dir = config.MELSPEC2SPEC_DIR / args.weights_dir
    config_path = experiment_dir / "config.json"
    out_hat_path = experiment_dir / 'gla_from_melspec.wav'
    metrics_path = experiment_dir / 'prediction_metrics.json'    
    audio_path = config.DATA_DIR / args.audio_path
    prediction_img_path = config.MELSPEC2SPEC_DIR / args.weights_dir / 'prediction.png'  

    if args.model_name != 'pinv':
        hparams = config.load_config(config_path)
    else:
        hparams = config.create_hparams()

    # Compute stft of example and then apply gla to retrieve the waveform back
    audio = open_audio(audio_path, hparams)
    stftspec = torch.abs(torch.as_tensor(librosa.stft(y=audio, 
                                                      n_fft=hparams.n_fft,
                                                      hop_length=hparams.hop_len)))

    stftspec_db_norm = normalize_db_spectr(to_db(stftspec)).float()

    if args.model_name == 'librosa':
        melfb = librosa.filters.mel(sr = hparams.sr, 
                                    n_fft = hparams.n_fft, 
                                    n_mels = hparams.n_mels)  
        melspec_db_norm = torch.matmul(torch.as_tensor(melfb), stftspec_db_norm)
        stftspec_hat_db_norm = librosa.feature.inverse.mel_to_stft(melspec_db_norm.numpy(), 
                                                                   sr = hparams.sr,
                                                                   n_fft = hparams.n_fft)
        stftspec_hat_db_norm = torch.as_tensor(stftspec_hat_db_norm)
    
    else:
        weights_dir = config.WEIGHTS_DIR / args.weights_dir
        model = build_model(hparams, args.model_name, weights_dir, args.best_weights)
        model.eval()
        melspec_db_norm = torch.matmul(model.pinvblock.melfb, stftspec_db_norm.to(hparams.device)).unsqueeze(0).unsqueeze(0)
        stftspec_hat_db_norm = model(melspec_db_norm).cpu()     
        
    # save audio
    stftspec_hat = to_linear(denormalize_db_spectr(stftspec_hat_db_norm))  
    out_hat, _ = fast_griffin_lim(torch.abs(stftspec_hat).cpu().detach().numpy().squeeze())
    sf.write(str(out_hat_path), out_hat, samplerate = hparams.sr)   
    # Compute out_hat 's spectrogram and compare it to the original spectrogram
    stftspec_gla_db_norm =  normalize_db_spectr(to_db(torch.abs(torch.as_tensor(librosa.stft(y=out_hat, 
                                                                                             n_fft=hparams.n_fft,
                                                                                             hop_length=hparams.hop_len)))))
    plot_prediction(denormalize_db_spectr(stftspec_db_norm).cpu().numpy().squeeze(), 
                    denormalize_db_spectr(stftspec_hat_db_norm).cpu().detach().numpy().squeeze(), 
                    hparams, 
                    prediction_img_path)
    
    metrics = {"mse": float(mse(stftspec_db_norm, stftspec_hat_db_norm)),
               "si-snr": float(si_snr_metric(stftspec_db_norm, stftspec_hat_db_norm)),
               "si-snr (after gla)": float(si_snr_metric(stftspec_db_norm, stftspec_gla_db_norm))}
    
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=4)

    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        choices = ["unet", "librosa", "convpinv", "pinv"],
                        type=str,
                        default = 'convpinv')
    parser.add_argument('--weights_dir',
                        type=str,
                        default='test')
    parser.add_argument('--best_weights',
                        type=bool,
                        help='if False loads the weights from the checkpoint',
                        default=True)
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args)