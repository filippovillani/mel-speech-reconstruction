import torch
import torch.nn as nn

from .layers import ConvDNN
from utils.utils import r2_to_c

class DeGLI(nn.Module):
    def __init__(self, hparams) -> None:
        
        super(DeGLI, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.degliblock = DeGLIBlock(hparams)
        
    def forward(self, x_n_stft_mag, x_stft_mag):
        """
        Args:
            x_n_stft_mag (torch.Tensor): input STFT spectrogram corrupted by noise
            [batch, 1, n_stft, n_frames]
            x_stft_mag (torch.Tensor): clean STFT spectrogram
        Returns:
            x_stft_hat_stack (torch.Tensor): stack of the output for each iteration of the algorithm
            [batch, 2, n_degli_repetitions, n_stft, n_frames]
        """
        x_stft_hat = self._initialize_stft(x_n_stft_mag)

        # for _ in range(self.hprms.n_degli_repetitions):
        x_stft_hat_stack = torch.stack([(x_stft_hat := self.degliblock(x_stft_hat, x_stft_mag)) for _ in range(self.hprms.n_degli_repetitions)], 
                                    dim=2)

        x_stft_hat_stack = self._magnitude_projection(x_stft_hat_stack, x_stft_mag)
        
        return x_stft_hat_stack
    
    def _magnitude_projection(self, x_hat_stft, x_stft_mag):
        
        x_stft_mag = x_stft_mag.unsqueeze(2).expand(-1,-1,x_hat_stft.shape[2],-1,-1)
        phase = torch.atan2(x_hat_stft[:,1], x_hat_stft[:,0]).unsqueeze(1)
        x_amp_proj = torch.cat([x_stft_mag * torch.cos(phase), x_stft_mag * torch.sin(phase)], axis=1)
            
        return x_amp_proj
    
    def _initialize_stft(self, x_stft_mag):
        # TODO: implement different phase initialization
        x_init_phase = (torch.rand(x_stft_mag.shape)*2*torch.pi-torch.pi).to(self.device)
        x_n_stft = torch.cat([x_stft_mag * torch.cos(x_init_phase), x_stft_mag * torch.sin(x_init_phase)], axis=1)   
        return x_n_stft
    
    
class DeGLIBlock(nn.Module):
    def __init__(self, hparams):
        
        super(DeGLIBlock, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.convdnn = ConvDNN(hparams) # here there are the only trainable parameters
    
    def forward(self, x_hat_stft, x_stft_mag):
        """_summary_

        Args:
            x_stft_mag (torch.Tensor): reference amplitude
            shape=[batch_size, 1, n_stft, n_frames] (1 channel for the real part, one for the imaginary).
            
            x_n_stft (torch.Tensor): short-time fourier transform of the signal whose phase is to be reconstructed.
            It is None just for the first block (in this case x_n_stft=x_stft_mag)  
            shape=[batch_size, 2, n_stft, n_frames] (1 channel for the real part, one for the imaginary).

        Returns:
            _type_: _description_
        """
        
        x_amp_proj = self._magnitude_projection(x_hat_stft, x_stft_mag)
        x_cons_proj = self._consistency_projection(x_amp_proj)
        x_est_residual = self.convdnn(x_hat_stft, x_amp_proj, x_cons_proj)
        
        x_hat_stft = x_cons_proj - x_est_residual
        
        return x_hat_stft
        
    def _magnitude_projection(self, x_hat_stft, x_stft_mag):
        
        phase = torch.atan2(x_hat_stft[:,1], x_hat_stft[:,0]).unsqueeze(1)
        x_amp_proj = torch.cat([x_stft_mag * torch.cos(phase), x_stft_mag * torch.sin(phase)], axis=1)
            
        return x_amp_proj
    
    def _consistency_projection(self, x_amp_proj):

        x_cons_proj = torch.istft(r2_to_c(x_amp_proj), n_fft=self.hprms.n_fft)    # G+ x
        x_cons_proj = torch.stft(x_cons_proj, n_fft=self.hprms.n_fft, return_complex=False).permute(0,3,1,2) # G G+ x 
       
        return x_cons_proj.float()
    
if __name__ == "__main__":
    from argparse import Namespace
    hparams = Namespace(n_fft = 1024,
                        hidden_channel = 32)
    x_stft_mag = torch.load(r'D:\GitHub_Portfolio\PhaseReconstruction\data\spectrograms\validation\SX445.WAV.wav.pt')
    x_stft_mag = x_stft_mag.unsqueeze(0).unsqueeze(0).float()
    model = DeGLI(hparams).float()
    x_n_stft = model(x_stft_mag)
