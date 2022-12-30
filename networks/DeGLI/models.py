import torch
import torch.nn as nn

from .layers import ConvDNN

class DeGLI(nn.Module):
    def __init__(self, hparams) -> None:
        
        super(DeGLI, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.degliblocks = nn.ModuleList([DeGLIBlock(hparams) for _ in range(10)])
        self.degliblock = DeGLIBlock(hparams)
        
    def forward(self, x_spectr):
        x_stft_hat = self._initialize_stft(x_spectr)
        # for block in self.degliblocks:
        #     x_stft_hat = block(x_spectr, x_stft_hat)
        for _ in range(self.hprms.n_degli_blocks):
            x_stft_hat = self.degliblock(x_spectr, x_stft_hat)
        x_wav_hat = torch.istft(self._r2_to_c(x_stft_hat), n_fft=self.hprms.n_fft)
        x_wav_hat = (x_wav_hat - torch.min(x_wav_hat)) / (torch.max(x_wav_hat) - torch.min(x_wav_hat))
        return x_stft_hat, x_wav_hat

    def _initialize_stft(self, stftspec):
        # TODO: implement different phase initialization
        x_init_phase = (torch.rand(stftspec.shape)*2*torch.pi-torch.pi).to(self.hprms.device)
        x_stft = torch.cat([stftspec * torch.cos(x_init_phase), stftspec * torch.sin(x_init_phase)], axis=1)   
        return x_stft
    
    def _r2_to_c(self, x_r2):
        
        x_re = x_r2[:,0,:,:]
        x_im = x_r2[:,1,:,:]
        x_c = x_re + 1j * x_im 

        return x_c
    
    
class DeGLIBlock(nn.Module):
    def __init__(self, hparams):
        
        super(DeGLIBlock, self).__init__()
        self.hprms = hparams
        self.device = hparams.device
        self.convdnn = ConvDNN(hparams) # here there are the only trainable parameters
    
    def forward(self, stftspec, x_stft):
        """_summary_

        Args:
            stftspec (torch.Tensor): reference amplitude
            shape=[batch_size, 1, n_stft, n_frames] (1 channel for the real part, one for the imaginary).
            
            x_stft (torch.Tensor): short-time fourier transform of the signal whose phase is to be reconstructed.
            It is None just for the first block (in this case x_stft=stftspec)  
            shape=[batch_size, 2, n_stft, n_frames] (1 channel for the real part, one for the imaginary).

        Returns:
            _type_: _description_
        """
            
        x_amp_proj = self._amplitude_projection(x_stft, stftspec)
        x_cons_proj = self._consistency_projection(x_amp_proj)
        x_est_residual = self.convdnn(x_stft, x_amp_proj, x_cons_proj)
        
        x_stft_hat = x_cons_proj - x_est_residual
        
        return x_stft_hat
        
    def _amplitude_projection(self, x_stft, stftspec):
        
        phase = torch.atan2(x_stft[:,0], x_stft[:,1]).unsqueeze(1)
        x_amp_proj = torch.cat([stftspec * torch.cos(phase), stftspec * torch.cos(phase)], axis=1)
            
        return x_amp_proj
    
    def _consistency_projection(self, x_amp_proj):

        x_cons_proj = torch.istft(self._r2_to_c(x_amp_proj), n_fft=self.hprms.n_fft)    # G+ x
        x_cons_proj = torch.stft(x_cons_proj, n_fft=self.hprms.n_fft).permute(0,3,1,2) # G G+ x 
       
        return x_cons_proj
    
    def _r2_to_c(self, x_r2):
        
        x_re = x_r2[:,0,:,:]
        x_im = x_r2[:,1,:,:]
        x_c = x_re + 1j * x_im 

        return x_c
    
if __name__ == "__main__":
    from argparse import Namespace
    hparams = Namespace(n_fft = 1024,
                        hidden_channel = 32)
    stftspec = torch.load(r'D:\GitHub_Portfolio\PhaseReconstruction\data\spectrograms\validation\SX445.WAV.wav.pt')
    stftspec = stftspec.unsqueeze(0).unsqueeze(0).float()
    model = DeGLIBlock(hparams).float()
    x_stft = model(stftspec)
    x_stft = model(stftspec, x_stft)