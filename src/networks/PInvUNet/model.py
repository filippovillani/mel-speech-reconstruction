import torch.nn as nn

from networks.PInvConv.layers import PInvBlock
from networks.UNet.models import UNet

class PInvUNet(nn.Module):
    def __init__(self, hparams):
        super(PInvUNet, self).__init__()
        
        self.pinv = PInvBlock(hparams)
        self.unet = UNet(hparams)
        
    def forward(self, x_melspec):
        
        x_stft_hat = self.pinv(x_melspec)
        x_stft_hat = self.unet(x_stft_hat)
        return x_stft_hat