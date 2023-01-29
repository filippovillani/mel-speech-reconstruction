import json

import torch

def save_to_json(data, paths):
    if not isinstance(data, list):
        data = [data]
    if not isinstance(paths, list):
        paths = [paths]
    for dat, path in zip(data, paths):
        with open(path, "w") as fp:
            json.dump(dat, fp, indent=4)

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