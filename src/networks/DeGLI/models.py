import torch
import torch.nn as nn

from .layers import AIGC, LastConv


class DeGLI(nn.Module):
    def __init__(self, hparams):
        super(DeGLI, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.degliblock = DeGLIBlock(hparams)
        self.repetitions = 1 # default value for training, changed for evaluation and test
        
    def forward(self, x_noisy, x_stft_mag):
        
        for _ in range(self.repetitions):
            x_noisy = self.degliblock(x_noisy, x_stft_mag)
        
        out = self.degliblock.magnitude_projection(x_noisy, x_stft_mag)
        return out
    
    
class DeGLIBlock(nn.Module):
    def __init__(self, hparams):
        super(DeGLIBlock, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.convdnn = AIGCCNN(hparams) # here there are the only trainable parameters
    
    def forward(self, x_hat_stft, x_stft_mag):
        
        x_amp_proj = self.magnitude_projection(x_hat_stft, x_stft_mag)
        x_cons_proj = self._consistency_projection(x_amp_proj)
        
        x_cat = torch.stack([x_hat_stft, x_amp_proj, x_cons_proj], axis=1)
                
        x_est_residual = self.convdnn(x_cat, x_stft_mag).squeeze()
        
        x_hat_stft = x_cons_proj - x_est_residual
        
        return x_hat_stft
        
    def magnitude_projection(self, x_hat_stft, x_stft_mag):
        
        phase = x_hat_stft.angle()
        x_amp_proj = x_stft_mag * torch.exp(1j * phase)
            
        return x_amp_proj
    
    def _consistency_projection(self, x_amp_proj):

        x_cons_proj = torch.istft(x_amp_proj, 
                                  n_fft = self.hprms.n_fft,
                                  window = torch.hann_window(self.hprms.n_fft).to(x_amp_proj.device)) # G+ x
        x_cons_proj = torch.stft(x_cons_proj, 
                                 n_fft = self.hprms.n_fft,
                                 window = torch.hann_window(self.hprms.n_fft).to(x_cons_proj.device), 
                                 return_complex = True) # G G+ x 
       
        return x_cons_proj

class AIGCCNN(nn.Module):
    def __init__(self, hparams):
        super(AIGCCNN, self).__init__()
        self.aigcc = nn.Sequential(AIGC(in_channels = 3,
                                         out_channels = hparams.degli_hidden_channels,
                                         kernel_size = hparams.degli_kernel_size),
                                   AIGC(in_channels = hparams.degli_hidden_channels,
                                         out_channels = hparams.degli_hidden_channels,
                                         kernel_size = hparams.degli_kernel_size),
                                   AIGC(in_channels = hparams.degli_hidden_channels,
                                        out_channels = hparams.degli_hidden_channels,
                                        kernel_size = hparams.degli_kernel_size))
    
        self.conv = LastConv(in_channels = hparams.degli_hidden_channels,
                             out_channels = 1,
                             kernel_size = (1,1))
        
    def forward(self, x_cat, x_stft_mag):
        
        for block in self.aigcc:
            x_cat = block(x_cat, x_stft_mag)
        x_cat = self.conv(x_cat)
        
        return x_cat
