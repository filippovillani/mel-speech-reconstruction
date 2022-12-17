import torch
import librosa
import soundfile as sf
import numpy as np
import argparse

from model import UNet
from layers import PInvBlock
from griffinlim import fast_griffin_lim
from plots import plot_prediction
from audioutils import open_audio, to_db, normalize_db_spectr, denormalize_db_spectr, to_linear, normalize_db_spectr
import config



def predict(args, hparams):
    experiment_dir = config.MELSPEC2SPEC_DIR / args.experiment_name
    weights_path = config.WEIGHTS_DIR / args.experiment_name / 'ckpt_weights'

    audio_path = config.DATA_DIR / args.audio_path
    out_hat_path = experiment_dir / 'gla_from_melspec.wav'
    # Compute stft of example and then apply gla to retrieve the waveform back
    audio = open_audio(audio_path, hparams)
    stftspec = torch.abs(torch.as_tensor(librosa.stft(y=audio, 
                                   n_fft=hparams.n_fft,
                                   hop_length=hparams.hop_len)))

    stftspec_db_norm = normalize_db_spectr(to_db(stftspec))
    # Instatiate the model
    # model = UNet(hparams).float().to(config.DEVICE)
    model = PInvBlock(hparams)
    model.eval()
    # model.load_state_dict(torch.load(weights_path))
    
    # Compute melspectrogram of example
    # melspec_db_norm = torch.matmul(model.pinvblock.melfb, torch.as_tensor(stftspec_db_norm).float().to(config.DEVICE))
    melspec_db_norm = torch.matmul(model.melfb, torch.as_tensor(stftspec_db_norm).float().to(config.DEVICE))
    stftspec_hat_db_norm = model(melspec_db_norm.unsqueeze(0).unsqueeze(0))
    
    # just save audio
    # stftspec_hat = to_linear(denormalize_db_spectr(stftspec_hat_db_norm))  
    # out_hat, _ = fast_griffin_lim(np.abs(stftspec_hat.cpu().detach().numpy().squeeze()))
    # sf.write(str(out_hat_path), out_hat, samplerate = hparams.sr)   
    

      
    plot_prediction(denormalize_db_spectr(stftspec_db_norm).cpu().numpy().squeeze(), 
                    stftspec_hat_db_norm.cpu().detach().numpy().squeeze(), 
                    hparams, 
                    args.experiment_name)
    
        
if __name__ == "__main__":
    hparams = config.create_hparams()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        type=str,
                        default='06_unet01')
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args, hparams)