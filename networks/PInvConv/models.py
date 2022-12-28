import torch
import torch.nn as nn

from .layers import PInvBlock, ConvBlock


class PInv(nn.Module):
    def __init__(self, hparams):

        super(PInv, self).__init__()
        self.device = hparams.device
        self.pinvblock = PInvBlock(hparams) 
        
    def forward(self, melspec):
        
        x = self.pinvblock(melspec)
        stft_hat = x / torch.max(x)
        return stft_hat
    
class PInvConv(nn.Module):
    def __init__(self, hparams):

        super(PInvConv, self).__init__()
        self.device = hparams.device

        in_channels = [1] + hparams.conv_channels + hparams.conv_channels[-2::-1]
        out_channels = in_channels[1:] + [in_channels[0]]
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.kernel_size) for l in range(len(in_channels))])
        

    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]

        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)
        for l in range(len(self.convblocks)):
            x = self.convblocks[l](x)
        
        x_max = torch.as_tensor([torch.max(x[q]) for q in range(x.shape[0])])
        stft_hat = torch.empty(x.shape)
        for b in range(stft_hat.shape[0]):
            stft_hat[b] = x[b] / x_max[b]
        stft_hat = stft_hat.to(x.device)
        return stft_hat
