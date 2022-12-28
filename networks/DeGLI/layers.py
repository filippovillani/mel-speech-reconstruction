import torch
import torch.nn as nn

class DeGLIBlock(nn.Module):
    def __init__(self, hparams):
        
        super(DeGLIBlock, self).__init__()
        self.hprms = hparams
        self.convdnn = ConvDNN(hparams)
    
    def _amplitude_projection(self, stftspec, x_stft):
        
        x_stft = stftspec * x_stft / (torch.abs(x_stft) + 1e-12)
            
        return x_stft
    
    def _consistency_projection(self, x_amp_proj):

        x_cons_proj = torch.istft(x_amp_proj.permute(0,2,3,1), n_fft=self.hprms.n_fft)    # G+ x
        x_cons_proj = torch.stft(x_cons_proj, n_fft=self.hprms.n_fft).permute(0,3,1,2) # G G+ x 
       
        return x_cons_proj
    
    def forward(self, stftspec, x_stft = None):
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
        if x_stft is None:
            # TODO: implement different phase initialization
            x_init_phase = torch.zeros(stftspec.shape)
            x_stft = torch.cat([stftspec * torch.cos(x_init_phase), stftspec * torch.sin(x_init_phase)], axis=1)
            
        x_amp_proj = self._amplitude_projection(stftspec, x_stft)
        x_cons_proj = self._consistency_projection(x_amp_proj)
        x_est_residual = self.convdnn(x_stft, x_amp_proj, x_cons_proj)
        
        out = x_cons_proj - x_est_residual
        
        return out
    
class ConvDNN(nn.Module):  
    def __init__(self, hparams):
        
        super(ConvDNN, self).__init__()
        self.convblock1 = ConvBlock(in_channels = 6,
                                    out_channels = hparams.hidden_channel,
                                    kernel_size = (11, 11))
        self.convblock23 = nn.Sequential(ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = (7, 3)),
                                         ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = (7, 3)))
        self.convblock45 = nn.Sequential(ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = (7, 3)),  
                                         ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = 2,
                                                   kernel_size = (7, 3),
                                                   last_block = True))
        
    def forward(self, x, x_amp_proj, x_cons_proj):
        
        x = torch.cat([x, x_amp_proj, x_cons_proj], axis=1) 
        x = self.convblock1(x)
        x = self.convblock23(x) + x
        x = self.convblock45(x)
        
        return x
            
        


class ConvBlock(nn.Module):  
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 last_block = False):
        
        super(ConvBlock, self).__init__()
        self.last_block = last_block
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = "same")
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.glu = nn.GLU(dim=1)
        
    def forward(self, x):
        
        x = self.conv(x)
        if not self.last_block:
            x = self.bn(x)
            x = self.glu(x)
        
        return x
        
if __name__ == "__main__":
    from argparse import Namespace
    hparams = Namespace(n_fft = 1024,
                        hidden_channel = 32)
    stftspec = torch.load(r'D:\GitHub_Portfolio\PhaseReconstruction\data\spectrograms\validation\SX445.WAV.wav.pt')
    stftspec = stftspec.unsqueeze(0).unsqueeze(0).float()
    model = DeGLIBlock(hparams).float()
    x_stft = model(stftspec)
    x_stft = model(stftspec, x_stft)