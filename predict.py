import torch
import librosa
import soundfile as sf
import numpy as np
import argparse

from model import Network
from griffinlim import fast_griffin_lim
from audioutils import melspectrogram, spectrogram
from plots import plot_prediction
import config

def open_audio(audio_path, hparams):
    # Open audio  
    audio, _ = librosa.load(audio_path, sr=hparams.sr)
    
    # pad or trunc all vectors to the same size
    if len(audio) < hparams.audio_len:
        pad_begin_len = np.random.randint(0, hparams.audio_len - len(audio))
        pad_end_len = hparams.audio_len - len(audio) - pad_begin_len
        
        pad_begin = np.zeros(pad_begin_len)
        pad_end = np.zeros(pad_end_len)
    
        audio = np.concatenate((pad_begin, audio, pad_end))  
        
    else:
        start_position = np.random.randint(0, len(audio) - hparams.audio_len)
        end_position = start_position + hparams.audio_len
        audio = audio[start_position:end_position]
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-12)
    return audio

def predict(args, hparams):
    experiment_dir = config.MELSPEC2SPEC_DIR / args.experiment_name
    weights_path = config.WEIGHTS_DIR / args.experiment_name / 'weights'

    audio_path = config.DATA_DIR / args.audio_path
    out_path = experiment_dir / 'gla_from_stftspec.wav' 
    out_hat_path = experiment_dir / 'gla_from_melspec.wav'

    # Compute stft of example and then apply gla to retrieve the waveform back
    audio = open_audio(audio_path, hparams)
    stftspec = np.abs(librosa.stft(y=audio, 
                                   n_fft=hparams.n_fft,
                                   hop_length=hparams.hop_len))
    
    out, _ = fast_griffin_lim(stftspec)
    sf.write(str(out_path), out, samplerate = hparams.sr)

    # Instatiate the model
    model = Network(hparams).float().to(config.DEVICE)
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    
    # Compute melspectrogram of example, then stft through NN and then apply gla
    melspec = torch.matmul(model.melfb, torch.as_tensor(stftspec).float().to(config.DEVICE))
    
    stftspec_hat = model.compute_stft_spectrogram(melspec.unsqueeze(0))
    melspec_hat = model.compute_mel_spectrogram(stftspec_hat.squeeze())
    out_hat, _ = fast_griffin_lim(np.abs(stftspec_hat.cpu().detach().numpy().squeeze()))
    sf.write(str(out_hat_path), out_hat, samplerate = hparams.sr)   
    
    plot_prediction(melspec.cpu().numpy(), 
                    melspec_hat.cpu().detach().numpy(), 
                    hparams, 
                    args.experiment_name)
    
        
if __name__ == "__main__":
    hparams = config.create_hparams()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        type=str,
                        default='prova03')
    parser.add_argument('--weights_dir',
                        type=str,
                        default=None)
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args, hparams)