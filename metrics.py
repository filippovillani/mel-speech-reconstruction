import numpy as np
import config
import soundfile as sf 

def si_ssnr(s_target, s_hat): 
    """
    Compute the Scale-Invariant Signal to Noise Ratio on the STFT magnitude,
    based on '2013 - A fast griffin-lim algorithm' and on '2018 - SDR - half-baked or well done?'

    Args:
        s_target (np.ndarray): spectrogram of target signal
        s_hat (np.ndarray): spectrogram of reconstructed signal

    Returns:
        snr (float): SI-SSNR
    """
    
    s_hat = s_hat - np.mean(s_hat)
    s_target = s_target - np.mean(s_target)
      
    s_target_scale = np.sum(s_target*s_hat) / (np.sum(np.power(np.abs(s_target), 2)) + 1e-8)
    s_target = s_target_scale * s_target
    
    target_power = np.sum(np.power(np.abs(s_target),2))
    disturb_power = np.sum(np.power(np.abs(s_target-s_hat), 2))
    
    snr = (target_power / (disturb_power + 1e-8))
    snr = 10 * np.log10(snr)    
    return snr

def mse(s_target, s_hat):
    error = s_hat - s_target 
    mse = np.mean(error**2)
    return mse


