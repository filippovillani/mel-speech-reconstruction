import torch
import numpy as np
import librosa 

import config

def melspectrogram(audio: np.ndarray,
                   sr: int = 16000,
                   n_mels: int = 96, 
                   n_fft: int = 1024, 
                   hop_len: int = 256)->np.ndarray:
    
    melspectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio,
                                                                        sr = sr, 
                                                                        n_fft = n_fft, 
                                                                        hop_length = hop_len,
                                                                        n_mels = n_mels), ref=np.max, top_db=80.) / 80. + 1.
    
    
    return melspectrogram

def stft(audio: np.ndarray,
         n_fft: int = 1024,
         hop_len: int = 256)->np.ndarray:
    
    spectrogram = librosa.stft(y=audio, 
                               n_fft=n_fft,
                               hop_length=hop_len)
    
    return spectrogram

def compute_mel_spectrogram(stft, hparams):
    melfb = torch.as_tensor(librosa.filters.mel(sr=hparams.sr, 
                                                n_fft=hparams.n_fft, 
                                                n_mels = hparams.n_mels)).to(config.DEVICE)
    
    out = torch.empty((hparams.batch_size,
                       hparams.n_channels,
                       hparams.n_mels, 
                       hparams.n_frames))  
    
    for n in range(stft.shape[0]):
        inp = stft[n].squeeze()
        out[n] = torch.matmul(melfb, inp).unsqueeze(0)
    
    return out