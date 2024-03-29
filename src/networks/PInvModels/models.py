import torch
import torch.nn as nn

from .layers import PInvBlock, ConvBlock
from networks.UNet.models import UNet


class PInv(nn.Module):
    def __init__(self, hparams):

        super(PInv, self).__init__()
        self.device = hparams.device
        self.pinvblock = PInvBlock(hparams) 
        
    def forward(self, melspec):
        
        x = self.pinvblock(melspec)
        stft_hat = x / torch.max(x)
        return stft_hat


class PInvUNet(nn.Module):
    def __init__(self, hparams):
        super(PInvUNet, self).__init__()
        
        self.pinv = PInvBlock(hparams)
        self.unet = UNet(hparams)
        self.relu = nn.ReLU()
        
    def forward(self, x_melspec):
        
        x_stft_hat = self.pinv(x_melspec)
        x = self.unet(x_stft_hat)
        
        x_stft_hat = x_stft_hat - x
        
        return x_stft_hat

class PInvConv(nn.Module):
    def __init__(self, hparams):

        super(PInvConv, self).__init__()
        self.device = hparams.device

        out_channels = hparams.conv_channels + hparams.conv_channels[-1::-1] + [1]
        in_channels = out_channels[::-1]
        in_channels = in_channels[:((len(out_channels)//2 + 1))] + out_channels[(len(out_channels)//2):-1]
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.conv_kernel_size,
                                                   hparams.drop_rate) for l in range(len(in_channels))])
        self.convblocks[-1].drop_rate = 0.

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


class PInvConvRes(nn.Module):
    def __init__(self, hparams):

        super(PInvConvRes, self).__init__()
        self.device = hparams.device

        out_channels = hparams.conv_channels + hparams.conv_channels[-1::-1] + [1]
        in_channels = out_channels[::-1]        
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.conv_kernel_size,
                                                   hparams.drop_rate) for l in range(len(in_channels))])
        self.convblocks[-1].drop_rate = 0.
        self.n_blocks = len(self.convblocks)
        
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]
        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)

        # Encoder
        x_res = []
        for l in range(self.n_blocks//2):
            x = self.convblocks[l](x)
            x_res.append(x)
        x_res = x_res[::-1]
        
        x = self.convblocks[self.n_blocks//2](x)
        # Decoder
        for l in range(self.n_blocks//2+1, self.n_blocks):
            x += x_res[l-(self.n_blocks//2+1)]
            x = self.convblocks[l](x)

        x_max = torch.as_tensor([torch.max(x[q]) for q in range(x.shape[0])])
        stft_hat = torch.empty(x.shape)
        for b in range(stft_hat.shape[0]):
            stft_hat[b] = x[b] / x_max[b]
        stft_hat = stft_hat.to(x.device)
        return stft_hat
    
    
class PInvConvSkip(nn.Module):
    def __init__(self, hparams):

        super(PInvConvSkip, self).__init__()
        self.device = hparams.device

        out_channels = hparams.conv_channels + hparams.conv_channels[-1::-1] + [1]
        in_channels = out_channels[::-1]
        in_channels = in_channels[:((len(out_channels)//2 + 1))] + [x*2 for x in out_channels[(len(out_channels)//2):-1]]
        
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.conv_kernel_size,
                                                   hparams.drop_rate) for l in range(len(in_channels))])
        self.convblocks[-1].drop_rate = 0.
        self.n_blocks = len(self.convblocks)
        
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]
        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)

        # Encoder
        x_cat = []
        for l in range(self.n_blocks//2):
            x = self.convblocks[l](x)
            x_cat.append(x)
        x_cat = x_cat[::-1]
        
        x = self.convblocks[self.n_blocks//2](x)
        # Decoder
        for l in range(self.n_blocks//2+1, self.n_blocks):
            x = torch.cat([x, x_cat[l-(self.n_blocks//2+1)]], axis=1)
            x = self.convblocks[l](x)

        x_max = torch.as_tensor([torch.max(x[q]) for q in range(x.shape[0])])
        stft_hat = torch.empty(x.shape)
        for b in range(stft_hat.shape[0]):
            stft_hat[b] = x[b] / x_max[b]
        stft_hat = stft_hat.to(x.device)
        return stft_hat
    

class PInvConvSkipNoBottleneck(nn.Module):
    # Here i'm actually implementing skip connections and not residual
    def __init__(self, hparams):

        super(PInvConvSkipNoBottleneck, self).__init__()
        self.device = hparams.device

        out_channels = hparams.conv_channels + hparams.conv_channels[-2::-1] + [1]
        in_channels = out_channels[::-1]
        in_channels = in_channels[:((len(out_channels)//2))] + [x*2 for x in out_channels[(len(out_channels)//2-1):-1]]
        
        
        self.pinvblock = PInvBlock(hparams)
        self.convblocks = nn.ModuleList([ConvBlock(in_channels[l], 
                                                   out_channels[l], 
                                                   hparams.conv_kernel_size,
                                                   hparams.drop_rate) for l in range(len(in_channels))])
        self.convblocks[-1].drop_rate = 0.
        self.n_blocks = len(self.convblocks)
        
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]
        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)

        # Encoder
        x_cat = []
        for l in range(self.n_blocks//2):
            x = self.convblocks[l](x)
            x_cat.append(x)
        x_cat = x_cat[::-1]
        
        # x = self.convblocks[self.n_blocks//2](x)
        # Decoder
        for l in range(self.n_blocks//2, self.n_blocks):
            x = torch.cat([x, x_cat[l-(self.n_blocks//2)]], axis=1)
            x = self.convblocks[l](x)

        x_max = torch.as_tensor([torch.max(x[q]) for q in range(x.shape[0])])
        stft_hat = torch.empty(x.shape)
        for b in range(stft_hat.shape[0]):
            stft_hat[b] = x[b] / x_max[b]
        stft_hat = stft_hat.to(x.device)
        return stft_hat