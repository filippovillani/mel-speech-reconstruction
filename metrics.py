import numpy as np

def ssnr(s_hat, s):
    snr = float(np.mean(np.power(np.abs(s_hat), 2) / np.power(np.abs(s - s_hat), 2)))
    return snr