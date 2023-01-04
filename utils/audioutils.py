import numpy as np
import torch
import librosa 

def compute_wav(x_n_stft, n_fft):
    
    x_wav_hat = torch.stack([torch.istft(x_n_stft[b], n_fft=n_fft) for b in range(x_n_stft.shape[0])], dim=0)
    x_wav_hat = min_max_normalization(x_wav_hat)
    return x_wav_hat
    
def to_db(spectrogram):
    spec_max = torch.max(spectrogram)
    spec_db = torch.clamp(20 * torch.log10(spectrogram / spec_max + 1e-12), min=-80, max=0)
    return spec_db

def to_linear(spectrogram_db):
    spec_lin = torch.pow(10, spectrogram_db / 20)
    return spec_lin

def normalize_db_spectr(spectrogram):
    return (spectrogram / 80) + 1

def denormalize_db_spectr(spectrogram):
    return (spectrogram - 1) * 80

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
    
def min_max_normalization(x_wav):
    if isinstance(x_wav, torch.Tensor):
        x_wav = (x_wav - torch.min(x_wav)) / (torch.max(x_wav) - torch.min(x_wav))
    if isinstance(x_wav, np.ndarray):
        x_wav = (x_wav - np.min(x_wav)) / (np.max(x_wav) - np.min(x_wav))
    return x_wav