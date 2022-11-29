import numpy as np
import config
import soundfile as sf 

def si_snr(s_target, s_hat): 
    s_hat = s_hat - np.mean(s_hat)
    s_target = s_target - np.mean(s_target)
      
    s_target_scale= np.dot(s_target, s_hat) / (np.sum(np.power(np.abs(s_target), 2)) + 1e-8)
    s_target = s_target_scale * s_target
    
    target_power = np.sum(np.power(np.abs(s_target),2))
    disturb_power = np.sum(np.power(np.abs(s_target-s_hat), 2))
    
    snr = (target_power / (disturb_power + 1e-8))
    snr = 10 * np.log10(snr)    
    return snr

if __name__ == "__main__":
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        clean, sr = sf.read(f)
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        estimated, _ = sf.read(f)
        
    snr = si_snr(clean, estimated)
