import torch
import numpy as torch
import torch


def griffin_lim(spectrogram: torch.Tensor, 
                n_iter: int = 500, 
                n_fft: int = 1024,
                init: str = "zeros"):
    
    X_init_phase = initialize_phase(spectrogram, init)
    
    X = spectrogram * torch.exp(1j * X_init_phase)
    for _ in range(n_iter):
        X_hat = torch.istft(X, n_fft=n_fft)    # G+ cn
        X_hat = torch.stft(X_hat, n_fft=n_fft, return_complex = True) # G G+ cn  
        X_phase = torch.angle(X_hat) 
        X = spectrogram * torch.exp(1j * X_phase)   # Pc1(Pc2(cn-1))  
    
    x = torch.istft(X, n_fft=n_fft)
    
    return x


def fast_griffin_lim(spectrogram: torch.Tensor,
                     n_iter: int = 500,
                     alpha: float = 0.99, 
                     n_fft: int = 1024,
                     init: str = "zeros"):

    X_init_phase = initialize_phase(spectrogram, init)
    
    # Initialize the algorithm
    X = spectrogram * torch.exp(1j * X_init_phase)
    prev_proj = torch.istft(X, n_fft=n_fft)
    prev_proj = torch.stft(prev_proj, n_fft=n_fft, return_complex = True)
    prev_proj_phase = torch.angle(prev_proj) 
    prev_proj = spectrogram * torch.exp(1j * prev_proj_phase) 
    
    for _ in range(n_iter+1):
        curr_proj = torch.istft(X, n_fft=n_fft)    # G+ cn            
        curr_proj = torch.stft(curr_proj, n_fft=n_fft, return_complex = True) # G G+ cn  

        curr_proj_phase = torch.angle(curr_proj) 
        curr_proj = spectrogram * torch.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))  
            
        X = curr_proj + alpha * (curr_proj - prev_proj)
        prev_proj = curr_proj

    x = torch.istft(X, n_fft=n_fft)

    return x

def initialize_phase(spectrogram, init = "zeros"):
    
    if init == "zeros":
        X_init_phase = torch.zeros_like(spectrogram)    
    elif init =="random":
        X_init_phase = torch.pi * (2 * torch.rand_like(spectrogram) - 1)
    else:
        raise ValueError(f"init must be 'zeros' or 'random', received: {init}")
    
    return X_init_phase