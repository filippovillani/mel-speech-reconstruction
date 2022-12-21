import torch
import librosa
import soundfile as sf
import argparse
import json

from model import build_model
from audioutils import open_audio, to_db, normalize_db_spectr, denormalize_db_spectr, to_linear, normalize_db_spectr
from griffinlim import fast_griffin_lim
from metrics import mse, si_snr_metric
from plots import plot_prediction
import config



def predict(hparams, args):
    
    experiment_dir = config.MELSPEC2SPEC_DIR / args.weights_dir
    out_hat_path = experiment_dir / 'gla_from_melspec.wav'
    metrics_path = experiment_dir / 'metrics.json'
    
    audio_path = config.DATA_DIR / args.audio_path

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
        model = build_model(hparams, args.model_name, args.weights_dir, args.best_weights)
        model.eval()
        melspec_db_norm = torch.matmul(model.pinvblock.melfb, stftspec_db_norm.to(config.DEVICE)).unsqueeze(0).unsqueeze(0)
        stftspec_hat_db_norm = model(melspec_db_norm).cpu()     
        
    # save audio
    stftspec_hat = to_linear(denormalize_db_spectr(stftspec_hat_db_norm))  
    out_hat, _ = fast_griffin_lim(torch.abs(stftspec_hat).cpu().detach().numpy().squeeze())
    sf.write(str(out_hat_path), out_hat, samplerate = hparams.sr)   

      
    plot_prediction(denormalize_db_spectr(stftspec_db_norm).cpu().numpy().squeeze(), 
                    denormalize_db_spectr(stftspec_hat_db_norm).cpu().detach().numpy().squeeze(), 
                    hparams, 
                    args.weights_dir)
    
    metrics = {"mse": float(mse(stftspec_db_norm, stftspec_hat_db_norm)),
               "si-snr": float(si_snr_metric(stftspec_db_norm, stftspec_hat_db_norm))}
    
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp)

    
if __name__ == "__main__":
    
    hparams = config.create_hparams()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir',
                        type=str,
                        default='convpinv50')
    parser.add_argument('--best_weights',
                        type=bool,
                        help='if False loads the weights from the checkpoint',
                        default=True)
    parser.add_argument('--model_name',
                        choices = ["unet", "librosa", "convpinv"],
                        help = 'unet: evaluates unet; librosa: evaluates librosa.feature.inverse.mel_to_stft()',
                        type=str,
                        default = 'convpinv')
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(hparams, args)