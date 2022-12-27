import torch
import torch.nn as nn

class DeGLIBlock(nn.Module):
    def __init__(self):
        
        super(DeGLIBlock, self).__init__()
        self.convdnn = ConvDNN()
        
    def _amplitude_projection(self, stftspec, x_stft = None):
        
        if x_stft is None:
            # TODO: implement different phase initialization
            x_init_phase = torch.zeros(stftspec.shape)
            x_stft = torch.cat([stftspec * torch.cos(x_init_phase), stftspec * torch.sin(x_init_phase)], axis=1)
            
        else:
            x_stft = stftspec * x_stft / (torch.abs(x_stft) + 1e-12)
            
        return x_stft
    
    def _consistency_projection(self):
        
        return
    
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
        x_amp_proj = self._amplitude_projection(stftspec, x_stft)
        return x_amp_proj
    
class ConvDNN(nn.Module):  
    def __init__(self):
        
        super(ConvDNN, self).__init__()
        ...
        
if __name__ == "__main__":
    stftspec = torch.load(r'D:\GitHub_Portfolio\PhaseReconstruction\data\spectrograms\validation\SX445.WAV.wav.pt')
    stftspec = stftspec.unsqueeze(0).unsqueeze(0)
    model = DeGLIBlock()
    x_stft = model(stftspec)
    x_stft = model(stftspec, x_stft)