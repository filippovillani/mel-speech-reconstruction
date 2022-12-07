import os
from pathlib import Path
from argparse import Namespace

def create_hparams():
    hparams = Namespace(batch_size = 4,
                        epochs = 50,
                        patience = 20,
                        lr = 1e-3,
                        sr = 16000,
                        n_mels = 96,
                        n_fft = 1024,
                        n_channels = 1,
                        hop_len = 256,
                        audio_ms = 4080,
                        min_noise_ms = 1000)
    
    audio_len_ = int(hparams.sr * hparams.audio_ms // 1000)
    frame_len_ = int(audio_len_ // hparams.hop_len + 1)
    hparams = Namespace(**vars(hparams),
                        audio_len = audio_len_,
                        frame_len = frame_len_)
    
    return hparams

MAIN_DIR = Path(__file__).parent
DATA_DIR = MAIN_DIR / "data"
SPECTR_DIR = DATA_DIR / "spectr"
MELSPECTR_DIR = DATA_DIR / "melspectr"
RESULTS_DIR = MAIN_DIR / "results"
GLA_RESULTS_DIR = RESULTS_DIR / "gla"
WINDOWS_IMG_DIR = RESULTS_DIR / "windows"

SEED = 42

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
if not os.path.exists(WINDOWS_IMG_DIR):
    os.mkdir(WINDOWS_IMG_DIR)  
    
if not os.path.exists(GLA_RESULTS_DIR):
    os.mkdir(GLA_RESULTS_DIR)
    
if not os.path.exists(SPECTR_DIR):
    os.mkdir(SPECTR_DIR)

if not os.path.exists(MELSPECTR_DIR):
    os.mkdir(MELSPECTR_DIR)