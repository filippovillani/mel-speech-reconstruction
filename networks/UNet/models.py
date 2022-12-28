import torch.nn as nn

from .layers import ContractingBlock, ExpandingBlock, PInvBlock, OutBlock

class UNet(nn.Module):
    def __init__(self, hparams):
        
        super(UNet, self).__init__()
        self.device = hparams.device

        self.pinvblock = PInvBlock(hparams)
        self.contrblock1 = ContractingBlock(in_channels = hparams.n_channels,
                                            out_channels = hparams.first_unet_channel_units,
                                            kernel_size = hparams.kernel_size)
        self.contrblock2 = ContractingBlock(in_channels = hparams.first_unet_channel_units,
                                            kernel_size = hparams.kernel_size)

        self.contrblock3 = ContractingBlock(in_channels = hparams.first_unet_channel_units * 2,
                                            kernel_size = hparams.kernel_size,
                                            last_block = True)
        self.contrblock4 = ContractingBlock(in_channels = hparams.first_unet_channel_units * 4,
                                            kernel_size = hparams.kernel_size,
                                            last_block = True)

        self.expandblock3 = ExpandingBlock(in_channels = hparams.first_unet_channel_units * 8,
                                           kernel_size = hparams.kernel_size)
        self.expandblock2 = ExpandingBlock(in_channels = hparams.first_unet_channel_units * 4,
                                           kernel_size = hparams.kernel_size)
        self.expandblock1 = ExpandingBlock(in_channels = hparams.first_unet_channel_units * 2,
                                           kernel_size = hparams.kernel_size,
                                           last_block = True)
        self.outblock = OutBlock(in_channels = hparams.first_unet_channel_units)
        
    def forward(self, melspec):
        stft_hat = self.pinvblock(melspec)
        x, x_cat1 = self.contrblock1(stft_hat)
        x, x_cat2 = self.contrblock2(x)
        x, _ = self.contrblock3(x)
        x = self.expandblock2(x, x_cat2)
        x = self.expandblock1(x, x_cat1)
        out = self.outblock(x)
        
        return out  