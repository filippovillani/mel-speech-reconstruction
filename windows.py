import numpy as np

# add a method that adds one sample if needed by DFT
def boxcar(N: int):
    window = np.ones(N)
    return window

def bartlett(N: int):
    if N % 2 == 0:
        N -= 1

    n = np.arange(N) - (N-1)/2
    window = (N - np.abs(n)) / N
    
    return window

def hann(N: int):
    n = np.arange(N) - (N-1)/2
    window = (0.5 + 0.5 * np.cos(np.pi * n / N))
    
    return window        

def hamming(N: int):
    n = np.arange(N) - (N-1)/2
    window = (0.54 + 0.46 * np.cos(np.pi * n / N))
    
    return window        
    