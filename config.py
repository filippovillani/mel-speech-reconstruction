import os
import numpy as np
import random
import torch
import json
from pathlib import Path
from argparse import Namespace

def create_hparams():   # training hparams
    hparams = Namespace(batch_size = 2,
                        epochs = 70,
                        patience = 20,
                        # audio and dataset hparams
                        lr = 1e-3,
                        sr = 16000,
                        n_mels = 96,
                        n_fft = 1024,
                        n_channels = 1,
                        hop_len = 256,
                        audio_ms = 4080,
                        min_noise_ms = 1000,
                        num_workers = 0,
                        # model hparams
                        first_channel_units = 64,
                        kernel_size = (3, 1))
    
    audio_len_ = int(hparams.sr * hparams.audio_ms // 1000)
    n_frames_ = int(audio_len_ // hparams.hop_len + 1)
    n_stft_ = int(hparams.n_fft//2 + 1)
    hparams = Namespace(**vars(hparams),
                        audio_len = audio_len_,
                        n_frames = n_frames_,
                        n_stft = n_stft_)
    
    return hparams

def save_config(config_path):
    
    hparams = vars(create_hparams())
    
    with open(config_path, "w") as fp:
        json.dump(hparams, fp, indent=4)

def load_config(config_path):
    
    with open(config_path, "r") as fp:
        hparams = json.load(fp)
    hparams = Namespace(**hparams)
    return hparams

def set_seeds(seed = 42):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seeds(SEED)

MAIN_DIR = Path(__file__).parent

# Data
DATA_DIR = MAIN_DIR / "data"
SPECTR_DIR = DATA_DIR / "spectrograms"

# Model's weights
WEIGHTS_DIR = MAIN_DIR / "weights"

# Results
RESULTS_DIR = MAIN_DIR / "results"
GLA_RESULTS_DIR = RESULTS_DIR / "gla"
WINDOWS_IMG_DIR = RESULTS_DIR / "windows"
MELSPEC2SPEC_DIR = RESULTS_DIR / "melspec2spec"

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

if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

if not os.path.exists(MELSPEC2SPEC_DIR):
    os.mkdir(MELSPEC2SPEC_DIR)