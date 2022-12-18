import torch
import torch.nn as nn
from torchsummary import summary

from layers import ContractingBlock, ExpandingBlock, PInvBlock, OutBlock
import config

class UNet(nn.Module):
    def __init__(self, hparams):
        
        super(UNet, self).__init__()
        
        self.pinvblock = PInvBlock(hparams)
        self.contrblock1 = ContractingBlock(in_channels = 1,
                                            out_channels = hparams.unet_first_channels,
                                            kernel_size = 3)
        self.contrblock2 = ContractingBlock(in_channels = hparams.unet_first_channels,
                                            kernel_size = 3)
        self.contrblock3 = ContractingBlock(in_channels = hparams.unet_first_channels * 2,
                                            kernel_size = 3)
        self.contrblock4 = ContractingBlock(in_channels = hparams.unet_first_channels * 4,
                                            kernel_size = 3,
                                            last_block = True)
        self.expandblock3 = ExpandingBlock(in_channels = hparams.unet_first_channels * 16,
                                           kernel_size = 3)
        self.expandblock2 = ExpandingBlock(in_channels = hparams.unet_first_channels * 8,
                                           kernel_size = 3)
        self.expandblock1 = ExpandingBlock(in_channels = hparams.unet_first_channels * 4,
                                           kernel_size = 3)
        self.outblock = OutBlock(in_channels = hparams.unet_first_channels)
        
    def forward(self, melspec):
        stft_hat = self.pinvblock(melspec)
        x, x_cat1 = self.contrblock1(stft_hat)
        x, x_cat2 = self.contrblock2(x)
        x, x_cat3 = self.contrblock3(x)
        x, _ = self.contrblock4(x)
        x = self.expandblock3(x, x_cat3)
        x = self.expandblock2(x, x_cat2)
        x = self.expandblock1(x, x_cat1)
        out = self.outblock(x)
        
        return out
     

        
        
if __name__=="__main__":
    
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_channels, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = UNet(hparams).to(config.DEVICE)
    
    print(batch.shape)
    print(model(batch).shape)
    print(summary(model, batch))

    
