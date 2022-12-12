import numpy as np
import librosa 

def amplitude_to_db(spectrogram: np.ndarray)->np.ndarray:
    """
    Convert input spectrogram from amplitude to dB and then normalize it in [0, 1]

    Args:
        spectrogram (np.ndarray): spectrogram to be converted

    Returns:
        np.ndarray: converted spectrogram
    """
    return librosa.amplitude_to_db(spectrogram,
                                   ref=np.max, 
                                   top_db=80.) / 80. + 1.
    
