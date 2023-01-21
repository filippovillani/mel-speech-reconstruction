import torch

def mse(s_target: torch.Tensor,
        s_hat: torch.Tensor)->torch.Tensor:
    error = s_target - s_hat
    return torch.mean(torch.square(error))

def si_sdr_metric(s_target: torch.Tensor,
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

def si_nsr_loss(enhanced_speech: torch.Tensor, 
                clean_speech: torch.Tensor)->torch.Tensor:

    s_hat = (enhanced_speech - torch.mean(enhanced_speech))
    s_target = (clean_speech - torch.mean(clean_speech))
       
    s_target = torch.div(torch.mul(torch.sum(torch.mul(s_hat, s_target)), s_target),
                         torch.sum(torch.pow(s_target, 2)) + 1e-12)
    
    SI_NSR_linear = torch.divide(torch.sum(torch.pow(s_hat - s_target, 2)), torch.sum(torch.pow(s_target, 2)))
    si_nsr = torch.mul(torch.log10(SI_NSR_linear), 10.)
    return si_nsr
