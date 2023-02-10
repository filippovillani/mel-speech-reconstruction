import json
from argparse import Namespace

import torch

def load_json(path):
    
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def save_to_json(data, paths):
    
    if not isinstance(data, list):
        data = [data]
    if not isinstance(paths, list):
        paths = [paths]
    for dat, path in zip(data, paths):
        with open(path, "w") as fp:
            json.dump(dat, fp, indent=4)


def save_config(hparams, config_path):
        
    with open(config_path, "w") as fp:
        json.dump(vars(hparams), fp, indent=4)


def load_config(config_path):
    
    hparams = load_json(config_path)
    hparams = Namespace(**hparams)
    
    return hparams

def r2_to_c(x_r2):
    
    x_re = x_r2[:,0]
    x_im = x_r2[:,1]
    x_c = x_re + 1j * x_im 
    return x_c

def c_to_r2(x_c):
    
    x_mag = torch.abs(x_c).unsqueeze(1)
    x_phase = torch.angle(x_c).unsqueeze(1)
    x_re = torch.cat([x_mag * torch.cos(x_phase), x_mag * torch.sin(x_phase)], axis=1) 
    return x_re

def r2_to_mag_phase(x_r2):
    
    x_mag = torch.sqrt(x_r2[:,0]**2 + x_r2[:,1]**2)
    x_phase = torch.atan2(x_r2[:,0], x_r2[:,1])
    return x_mag, x_phase      