import librosa
import numpy as np
import soundfile as sf
import torch


def open_audio(audio_path, sr):

    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.as_tensor(audio)
    audio = standardization(audio)
    
    return audio


def compute_wav(x_stft, n_fft):

    if x_stft.dim() == 3:
        x_wav = torch.stack([torch.istft(x_stft[b], 
                                         n_fft = n_fft, 
                                         window = torch.hann_window(n_fft).to(x_stft[b].device)) for b in range(x_stft.shape[0])], dim=0)
    else:
        x_wav = torch.istft(x_stft, 
                            n_fft = n_fft,
                            window = torch.hann_window(n_fft).to(x_stft.device))
    x_wav = standardization(x_wav) 
    return x_wav


def save_audio(x_wav, x_wav_path, sr = 16000):
    
    if isinstance(x_wav, torch.Tensor):
        x_wav = x_wav.cpu().detach().numpy()
    x_wav = min_max_normalization(x_wav.squeeze())
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


def initialize_random_phase(x_stft_mag, init = "randn"):
    
    if init == "randn":
        phase = torch.randn_like(x_stft_mag)
    elif init == "zeros":
        phase = torch.zeros_like(x_stft_mag)
        
    x_stft = x_stft_mag * torch.exp(1j * phase)
    return x_stft    


def create_noise(signal, max_snr_db = 12, min_snr_db = -6):

    snr_db = (max_snr_db - min_snr_db) * torch.rand((1)) + min_snr_db
    snr = torch.pow(10, snr_db/10).to(signal.device)

    signal_power = torch.mean(torch.abs(signal) ** 2)
    
    noise_power = signal_power / (snr + 1e-12)
    noise = torch.sqrt(noise_power) * torch.randn_like(signal)
    
    return noise

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