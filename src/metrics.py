import torch
import torch.nn as nn

def mse(s_target: torch.Tensor,
        s_hat: torch.Tensor)->torch.Tensor:
    error = s_target - s_hat
    return torch.mean(torch.square(error))

def si_snr_metric(s_target: torch.Tensor,
                   s_hat: torch.Tensor)->torch.Tensor: 
    """
    Compute the Scale-Invariant Signal to Noise Ratio on the STFT magnitude,
    based on '2013 - A fast griffin-lim algorithm' and on '2018 - SDR - half-baked or well done?'
    Args:
        s_target (torch.Tensor): spectrogram of target signal
        s_hat (torch.Tensor): spectrogram of reconstructed signal
    Returns:
        snr (float): SI-SNR
    """
    # Zero-mean normalization
    s_hat = (s_hat - torch.mean(s_hat))
    s_target = (s_target - torch.mean(s_target))
       
    s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                         torch.sum(torch.pow(s_target, 2)) + 1e-12)
    
    e_noise = s_hat - s_target
    SI_SNR_linear = torch.divide(torch.sum(torch.pow(s_target, 2)), torch.sum(torch.pow(e_noise, 2))  + 1e-12)
    SI_SNR = torch.mul(torch.log10(SI_SNR_linear), 10.)
    return SI_SNR 

class SI_SDR(nn.Module):
    def __init__(self):
        super(SI_SDR, self).__init__()
    
    def forward(self, s_target, s_hat):
        return self._si_sdr_metric(s_target, s_hat)
    
    def _si_sdr_metric(self,
                       s_target: torch.Tensor,
                       s_hat: torch.Tensor)->torch.Tensor: 
        """
        Compute the Scale-Invariant Signal to Noise Ratio on the STFT magnitude,
        based on '2013 - A fast griffin-lim algorithm' and on '2018 - SDR - half-baked or well done?'

        Args:
            s_target (torch.Tensor): spectrogram of target signal
            s_hat (torch.Tensor): spectrogram of reconstructed signal

        Returns:
            sdr (float): SI-sdr
        """
        # Zero-mean normalization
        s_hat = (s_hat - torch.mean(s_hat))
        s_target = (s_target - torch.mean(s_target))
        
        s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                            torch.sum(torch.pow(s_target, 2)) + 1e-12)
        
        e_noise = s_hat - s_target
        SI_SDR_linear = torch.divide(torch.sum(torch.pow(s_target, 2)), torch.sum(torch.pow(e_noise, 2)) + 1e-12)
        si_sdr = torch.mul(torch.log10(SI_SDR_linear), 10.)
        return si_sdr 
