import json
import os
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


def create_hparams(model_name: str = None):   # training hparams
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == "unet":
        model_hparams = Namespace(first_unet_channel_units = 32)
    elif model_name == "convpinv":
        model_hparams = Namespace(conv_channels = [32, 64],
                                  kernel_size = 3)
    elif model_name == "degli" or model_name == "degliblock":
        model_hparams = Namespace(hidden_channel = 32,
                                  kernel_size = (3,3),
                                  n_degli_repetitions = 200)
    else:
        model_hparams = Namespace()
        
    hparams = Namespace(batch_size = 1,
                        lr = 1e-3,
                        epochs = 70,
                        patience = 20,
                        num_workers = 0,
                        device = device,
                        # audio hparams
                        sr = 16000,
                        n_mels = 96,
                        n_fft = 1024,
                        n_channels = 1,
                        hop_len = 256,
                        audio_ms = 1040,
                        audio_thresh = 0.05,
                        min_noise_ms = 1000)
    # more audio parameters
    audio_len_ = int(hparams.sr * hparams.audio_ms // 1000)
    n_frames_ = int(audio_len_ // hparams.hop_len + 1)
    n_stft_ = int(hparams.n_fft//2 + 1)
    
    hparams = Namespace(**vars(hparams),
                        **vars(model_hparams),
                        audio_len = audio_len_,
                        n_frames = n_frames_,
                        n_stft = n_stft_)
    
    return hparams

def save_config(hparams, config_path):
        
    with open(config_path, "w") as fp:
        json.dump(vars(hparams), fp, indent=4)

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

# For reproducibility
SEED = 42
set_seeds(SEED)

# Directories
MAIN_DIR = Path(__file__).parent

# Data
DATA_DIR = MAIN_DIR / "data"
STFT_DIR = DATA_DIR / "stft"

# Models' weights
WEIGHTS_DIR = MAIN_DIR / "weights"

# Results
RESULTS_DIR = MAIN_DIR / "results"
GLA_RESULTS_DIR = RESULTS_DIR / "gla"
WINDOWS_IMG_DIR = RESULTS_DIR / "windows"
MELSPEC2SPEC_DIR = RESULTS_DIR / "melspec2spec"
SPEC2WAV_DIR = RESULTS_DIR / "spec2wav"

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
if not os.path.exists(WINDOWS_IMG_DIR):
    os.mkdir(WINDOWS_IMG_DIR)  
    
if not os.path.exists(GLA_RESULTS_DIR):
    os.mkdir(GLA_RESULTS_DIR)
    
if not os.path.exists(STFT_DIR):
    os.mkdir(STFT_DIR)

if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

if not os.path.exists(MELSPEC2SPEC_DIR):
    os.mkdir(MELSPEC2SPEC_DIR)

if not os.path.exists(SPEC2WAV_DIR):
    os.mkdir(SPEC2WAV_DIR)