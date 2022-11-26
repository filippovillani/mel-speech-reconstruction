import numpy as np
import argparse
from scipy.fft import fft

from plots import plot_window

# add a method that adds one sample if needed by DFT
def boxcar(N: int):
    window = np.ones(N, dtype=float)
    window[0] = 0.5
    window[-1] = 0.5
    return window

def bartlett(N: int):
    n = np.arange(N) - (N//2-1)
    window = (N - np.abs(n)) / N
    
    return window

def hann(N: int):
    n = np.arange(N) - (N//2-1)
    window = (0.5 + 0.5 * np.cos(np.pi * n / N))
    return window

def hamming(N: int):
    n = np.arange(N) - (N//2-1)
    window = (0.54 + 0.46 * np.cos(np.pi * n / N))
    
    return window   

       
    
def main(args):
    T = 1 / args.sr
    
    if args.window == 'boxcar':
        window = boxcar(args.N)
    if args.window == 'bartlett':
        window = bartlett(args.N)
    if args.window == 'hann':
        window = hann(args.N)
    if args.window == 'hamming':
        window = hamming(args.N)
        
    window =np.concatenate([np.zeros(1024), window, np.zeros(1024)])
    spectrum = fft(window)
    plot_window(window, spectrum, args.window, args.N, T)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('window',
                        type=str,
                        choices=['boxcar', 'bartlett', 'hann', 'hamming'])
    parser.add_argument('--N',
                        type=int,
                        default=512)
    parser.add_argument('--sr',
                        type=int,
                        default=16000)
    args = parser.parse_args()
    main(args)
    