import numpy as np
import librosa 

def to_db(spectrogram):
    spec_max = np.max(spectrogram)
    spec_db = np.clip(20 * np.log10(spectrogram / spec_max + 1e-12), a_min=-80, a_max=0)
    return spec_db

def to_linear(spectrogram_db):
    spec_lin = np.power(10, spectrogram_db / 20)
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
    
