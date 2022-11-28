import numpy as np

def ssnr(s_hat, s_target):
    s_target = np.divide(np.multiply(np.dot(s_hat, s_target), s_target), np.power(np.abs(s_hat),2))
    e_noise = s_hat - s_target
    snr = 10*np.log10(np.sum(np.power(np.abs(s_target), 2)) / np.sum(np.power(np.abs(e_noise), 2)))
    return snr

def si_snr(s_target, s_hat):
    s_target = np.divide(np.multiply(np.dot(s_hat, s_target), s_target), np.power(np.abs(s_hat),2))
    s_targ_power = np.power(np.abs(s_target), 2)
    e_noise = s_target - s_hat
    e_noise_power = np.power(np.abs(e_noise), 2)
    snr = 10 * np.log(np.mean(np.divide(s_targ_power, e_noise_power)))
    return snr