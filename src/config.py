import json
import os
import sys
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

def create_hparams(model_name: str = None):   # training hparams
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    training_hparams = Namespace(batch_size = 1,
                                 lr = 1e-4,
                                 weights_decay = 1e-4,
                                 epochs = 70,
                                 patience = 20,
                                 lr_patience = 3,
                                 loss = "mse") # can be one of ["l1", "complexmse", "mse", "frobenius"]
                                 
    
    if model_name in ["unet", "pinvunet"]:
        model_hparams = Namespace(first_unet_channel_units = 32,
                                  kernel_size = (5,3),
                                  drop_rate = 0.1)
    elif model_name == "pinvconv":
        model_hparams = Namespace(conv_channels = [64, 32],
                                  kernel_size = (3,3),
                                  drop_rate = 0.)
    elif model_name == "degli":
        model_hparams = Namespace(hidden_channel = 32,
                                  kernel_size = (5,3),
                                  val_degli_rep = 3,
                                  degli_data_lr = 1e-6,
                                  max_snr_db = 12,
                                  min_snr_db = -6)
    else:
        model_hparams = Namespace()
    
    audio_hparams = Namespace(sr = 16000,
                              n_mels = 80,
                              n_fft = 1024,
                              n_channels = 1,
                              hop_len = 256,
                              audio_ms = 1040,
                              min_noise_ms = 1000)
    # Other useful audio parameters
    audio_len_ = int(audio_hparams.sr * audio_hparams.audio_ms // 1000)
    n_frames_ = int(audio_len_ // audio_hparams.hop_len + 1)
    n_stft_ = int(audio_hparams.n_fft//2 + 1)
    
    hparams = Namespace(**vars(training_hparams),
                        **vars(model_hparams),
                        device = device,
                        num_workers = 0,
                        **vars(audio_hparams),
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
MAIN_DIR = Path(__file__).parent.parent
sys.path.append(MAIN_DIR)

# Data
DATA_DIR = MAIN_DIR / "data"
STFT_DIR = DATA_DIR / "stft"

# Models' weights
WEIGHTS_DIR = MAIN_DIR / "weights"

# Results
RESULTS_DIR = MAIN_DIR / "results"
SPEC2WAV_DIR = RESULTS_DIR / "spec2wav"
MELSPEC2SPEC_DIR = RESULTS_DIR / "melspec2spec"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"

_dirs = [WEIGHTS_DIR, RESULTS_DIR, SPEC2WAV_DIR, MELSPEC2SPEC_DIR, COMPARISONS_DIR]

for dir in _dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)