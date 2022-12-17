import torch
import torch.nn as nn
from argparse import Namespace
import librosa 

import config

# TODO: 
# class PInvLayer(nn.Module):

import torch
import torch.nn as nn

class ContractingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 last_block = False):

        super(ContractingBlock, self).__init__()
        
        self.last_block = last_block
        out_channels = in_channels * 2 if in_channels != 1 else 64
        conv_cat_channels = out_channels * 2
        
        self.convC1 = nn.Conv2d(in_channels = in_channels, 
                                out_channels = out_channels, 
                                kernel_size = kernel_size, 
                                padding = 'same')
        nn.init.kaiming_normal_(self.convC1.weight)
        self.bnC1 = nn.BatchNorm2d(out_channels)        
        self.reluC1 = nn.ReLU() 
        self.convC2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        nn.init.kaiming_normal_(self.convC2.weight)
        self.bnC2 = nn.BatchNorm2d(out_channels)
        self.reluC2 = nn.ReLU() 
        self.dropC = nn.Dropout(0.3)
        self.poolC = nn.MaxPool2d(kernel_size=2)
        self.conv_cat = nn.Conv2d(in_channels = out_channels, 
                                out_channels = conv_cat_channels, 
                                kernel_size = kernel_size,
                                padding = 'same')

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): [batch_size, in_channels, n_mels, n_frames]

        Returns:
            out (torch.Tensor): 
                if max_pool: [batch_size, in_channels * 2, n_mels // 2, n_frames // 2]
                else:        [batch_size, in_channels * 2, n_mels, n_frames]
        """
        x = self.convC1(x)
        x = self.bnC1(x)        
        x = self.reluC1(x)        
        x = self.convC2(x)
        x = self.bnC2(x)
        x_cat = self.reluC2(x)        
        if self.last_block:
            out = x
            x_cat = None
        else:
            out = self.poolC(x_cat)
            x_cat = self.conv_cat(out)            
        return out, x_cat
    

class ExpandingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size):

        super(ExpandingBlock, self).__init__()
        mid_channels = in_channels // 2
        out_channels = in_channels // 4
        
        self.upsamp = nn.Upsample(scale_factor=2, 
                                mode='bilinear', 
                                align_corners=True)

        self.convE1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding='same')
            
        nn.init.kaiming_normal_(self.convE1.weight)
        self.bnE1 = nn.BatchNorm2d(mid_channels)
        self.reluE1 = nn.ReLU() 
        self.convE2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding='same')
        nn.init.kaiming_normal_(self.convE2.weight)
        self.bnE2 = nn.BatchNorm2d(out_channels)
        self.reluE2 = nn.ReLU() 
        self.dropE = nn.Dropout(0.3)
        
    def forward(self, x, x_cat):
        x = self.upsamp(x) 
        x = torch.cat((x, x_cat), dim=1) 
        x = self.convE1(x) 
        x = self.bnE1(x) 
        x = self.reluE1(x)
        x = self.convE2(x)
        x = self.bnE2(x)
        x = self.reluE2(x)
        out = self.dropE(x)
            
        return out
    

class OutBlock(nn.Module):
    def __init__(self,
                 in_channels):

        super(OutBlock, self).__init__()
        self.convOut1 = nn.Conv2d(in_channels = in_channels, 
                                  out_channels = in_channels//2, 
                                  kernel_size = 2,
                                  padding = 2)
        self.reluOut1 = nn.ReLU() 
        self.convOut2 = nn.Conv2d(in_channels = in_channels//2, 
                                  out_channels = 1,
                                  kernel_size = 1, 
                                  padding = 'same')
        self.reluOut2 = nn.ReLU() 
        
    def forward(self, x):
        
        x = self.convOut1(x)
        x = self.reluOut1(x)
        x = self.convOut2(x)
        out = self.reluOut2(x)
        
        return out

class PInvBlock(nn.Module):
    def __init__(self,
                 n_mels,
                 n_fft,
                 sr):

        super(PInvBlock, self).__init__()
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = sr, 
                                                         n_fft = n_fft, 
                                                         n_mels = n_mels)).to(config.DEVICE)
        
    
    def forward(self, melspec):
        
        stft_hat = torch.matmul(torch.linalg.pinv(self.melfb), melspec)
        return stft_hat
    

class UNet(nn.Module):
    def __init__(self, hparams):
        
        super(UNet, self).__init__()
        
        self.pinvblock = PInvBlock(n_mels=hparams.n_mels,
                                   n_fft = hparams.n_fft, 
                                   sr = hparams.sr)
        self.contrblock1 = ContractingBlock(in_channels = 1,
                                            kernel_size = 3)
        self.contrblock2 = ContractingBlock(in_channels = 64,
                                            kernel_size = 3)
        self.contrblock3 = ContractingBlock(in_channels = 128,
                                            kernel_size = 3,
                                            last_block = True)
        self.expandblock2 = ExpandingBlock(in_channels = 512,
                                           kernel_size = 3)
        self.expandblock1 = ExpandingBlock(in_channels = 256,
                                           kernel_size = 3)
        self.outblock = OutBlock(in_channels = 64)
        
    def forward(self, melspec):
        stft_hat = self.pinvblock(melspec)
        x, x_cat1 = self.contrblock1(stft_hat)
        x, x_cat2 = self.contrblock2(x)
        x, _ = self.contrblock3(x)
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
    
