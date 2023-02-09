import librosa
import numpy as np
import soundfile as sf
import torch


def open_audio(audio_path, sr, audio_len):

    audio, _ = librosa.load(audio_path, sr=sr)
    # pad_len = audio_len - (len(audio) % audio_len)
    # if pad_len != 0:
    #     pad = np.zeros((pad_len,))
    #     audio = np.concatenate((audio, pad))
    
    audio = torch.as_tensor(audio)
    audio = standardization(audio)
    
    return audio


def compute_wav(x_stft, n_fft):

    x_wav = torch.stack([torch.istft(x_stft[b], n_fft=n_fft) for b in range(x_stft.shape[0])], dim=0)
    x_wav = standardization(x_wav) 
    
    return x_wav


def save_audio(x_wav, x_wav_path, sr = 16000):
    if isinstance(x_wav, torch.Tensor):
        x_wav = x_wav.cpu().detach().numpy()
    x_wav = min_max_normalization(x_wav)
    sf.write(x_wav_path, x_wav, sr)


def to_db(spectrogram, power_spectr = False, min_db = -80):
    scale = 10 if power_spectr else 20
    spec_max = torch.max(spectrogram)
    spec_db = torch.clamp(scale * torch.log10(spectrogram / spec_max + 1e-12), min=min_db, max=0)
    return spec_db


def to_linear(spectrogram_db):
    spec_lin = torch.pow(10, spectrogram_db / 20)
    return spec_lin


def normalize_db_spectr(spectrogram):
    return (spectrogram / 80) + 1


def denormalize_db_spectr(spectrogram):
    return (spectrogram - 1) * 80


def pad_audio(audio, audio_len):
    
    pad_len = audio_len - len(audio) 
    pad = torch.zeros(pad_len)
    audio = torch.cat((audio, pad))
    return audio


def trunc_audio(audio, audio_len):
    
    pad_begin_len = np.random.randint(0, audio_len - len(audio))
    pad_end_len = audio_len - len(audio) - pad_begin_len
    
    pad_begin = np.zeros(pad_begin_len)
    pad_end = np.zeros(pad_end_len)

    audio = np.concatenate((pad_begin, audio, pad_end))    

    return audio


def segment_audio(audio_path, sr, audio_len):
    
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = standardization(torch.as_tensor(audio))
    audio_out = []
    mean = []
    std = []
    n = 0
    while len(audio) - n * audio_len > 0:
        audio_out.append(audio[n*audio_len:(n+1)*audio_len])
        mean.append(audio_out[n].mean())
        std.append(audio_out[n].std())
        n += 1
    audio_out[-1] = pad_audio(audio_out[-1], audio_len)
    audio_out = torch.stack(audio_out, dim=0)
    return audio_out, mean, std
    

def min_max_normalization(x_wav):
    if isinstance(x_wav, torch.Tensor):
        x_wav = (x_wav - torch.min(x_wav)) / (torch.max(x_wav) - torch.min(x_wav))
    if isinstance(x_wav, np.ndarray):
        x_wav = (x_wav - np.min(x_wav)) / (np.max(x_wav) - np.min(x_wav))
    return x_wav

def standardization(x_wav):
    return (x_wav - x_wav.mean()) / (x_wav.std() + 1e-12)

def set_mean_std(x_wav, mean, std):
    return ((x_wav - x_wav.mean()) * std / (x_wav.std() + 1e-12)) + mean