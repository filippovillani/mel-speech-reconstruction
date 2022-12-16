import torch
import librosa
import soundfile as sf
import numpy as np
import argparse

from model import MelSpect2Spec
from griffinlim import fast_griffin_lim
from plots import plot_prediction
from audioutils import open_audio, to_db, normalize_db_spectr, denormalize_db_spectr, to_linear
import config



def predict(args, hparams):
    experiment_dir = config.MELSPEC2SPEC_DIR / args.experiment_name
    weights_path = config.WEIGHTS_DIR / args.experiment_name / 'best_weights'

    audio_path = config.DATA_DIR / args.audio_path
    out_path = experiment_dir / 'gla_from_stftspec.wav' 
    out_hat_path = experiment_dir / 'gla_from_melspec.wav'
    out_pinv_path = experiment_dir / 'gla_from_pinvmelspec.wav'
    # Compute stft of example and then apply gla to retrieve the waveform back
    audio = open_audio(audio_path, hparams)
    stftspec = np.abs(librosa.stft(y=audio, 
                                   n_fft=hparams.n_fft,
                                   hop_length=hparams.hop_len))
    
    #out, _ = fast_griffin_lim(stftspec)
    #sf.write(str(out_path), out, samplerate = hparams.sr)

    # Instatiate the model
    model = MelSpect2Spec(hparams).float().to(config.DEVICE)
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    
    # Compute melspectrogram of example
    melspec = torch.matmul(model.melfb, torch.as_tensor(stftspec).float().to(config.DEVICE))
    melspec_db = normalize_db_spectr(to_db(melspec))
    
    stftspec_hat_db = model(melspec_db.unsqueeze(0).unsqueeze(0))
    stftspec_hat = to_linear(denormalize_db_spectr(stftspec_hat_db))  
    melspec_hat_db = to_db(torch.matmul(model.melfb, stftspec_hat.squeeze()))
    out_hat, _ = fast_griffin_lim(np.abs(stftspec_hat.cpu().detach().numpy().squeeze()))
    sf.write(str(out_hat_path), out_hat, samplerate = hparams.sr)   
    

      
    plot_prediction(melspec_db.cpu().numpy(), 
                    melspec_hat_db.cpu().detach().numpy(), 
                    hparams, 
                    args.experiment_name)
    
        
if __name__ == "__main__":
    hparams = config.create_hparams()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        type=str,
                        default='05_mse_db')
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args, hparams)