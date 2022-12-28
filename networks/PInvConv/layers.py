import torch
import torch.nn as nn
import librosa 

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = 'same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class PInvBlock(nn.Module):
    def __init__(self, hparams):

        super(PInvBlock, self).__init__()
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = hparams.sr, 
                                                         n_fft = hparams.n_fft, 
                                                         n_mels = hparams.n_mels)).to(hparams.device)
        
    
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]

        Returns:
            _type_: _description_
        """
        stft_hat = torch.clamp(torch.matmul(torch.linalg.pinv(self.melfb), melspec), min=0, max=1)
        
        return stft_hat
