import torch
import torch.nn as nn
import librosa 

import config


class ContractingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 out_channels = None,
                 last_block = False):

        super(ContractingBlock, self).__init__()
        
        self.last_block = last_block
        if out_channels is None and in_channels != 1:
            out_channels = in_channels * 2
        elif in_channels == 1 and out_channels is None:
            raise RuntimeError("If in_channels==1 you need to provide out_channels")
                    
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

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): [batch_size, in_channels, n_mels, n_frames]

        Returns:
            out (torch.Tensor):
                if in_channels == 1: in_channels = hparams.first_channel_units
                if last_block: [batch_size, in_channels * 2, n_mels, n_frames]
                else: [batch_size, in_channels * 2, n_mels // 2, n_frames // 2]
                
            x_cat (torch.Tensor):
                if last_block: x_cat = None
                else: [batch_size, in_channels * 2, n_mels, n_frames]
        """
        x = self.convC1(x)
        x = self.bnC1(x)        
        x = self.reluC1(x)        
        x = self.convC2(x)
        x = self.bnC2(x)
        x_cat = self.reluC2(x)        
        if self.last_block:
            out = x_cat
            x_cat = None
        else:
            out = self.poolC(x_cat)
        return out, x_cat
    

class ExpandingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 last_block = False):

        super(ExpandingBlock, self).__init__()
        
        out_channels = in_channels // 2
        if last_block:
            self.upconv = nn.Conv2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = (4,3),
                                    padding = (2,1))
        else:
            self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.upsamp = nn.Upsample(scale_factor=2, 
                                mode='bilinear', 
                                align_corners=True)

        self.convE1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
            
        nn.init.kaiming_normal_(self.convE1.weight)
        self.bnE1 = nn.BatchNorm2d(out_channels)
        self.reluE1 = nn.ReLU() 
        self.convE2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        nn.init.kaiming_normal_(self.convE2.weight)
        self.bnE2 = nn.BatchNorm2d(out_channels)
        self.reluE2 = nn.ReLU() 
        self.dropE = nn.Dropout(0.3)
        
    def forward(self, x, x_cat): 
        """ 
        Args:
            x (torch.Tensor): [batch_size, in_channels, n_mels // 2, n_frames // 2]
            x_cat (torch.Tensor): [batch_size, in_channels // 2, n_mels, n_frames]
        Returns:
            out (torch.Tensor):
                if last_block:  [batch_size, in_channels + 1, n_mels, n_frames]
                else: [batch_size, in_channels, n_mels, n_frames]       
            
        """
        x = self.upsamp(x)
        x = self.upconv(x)
        x = torch.cat((x, x_cat), axis=1) 
        
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
                                  kernel_size = 3,
                                  padding = 'same')
        self.reluOut1 = nn.ReLU() 
        self.convOut2 = nn.Conv2d(in_channels = in_channels//2, 
                                  out_channels = 1,
                                  kernel_size = 1, 
                                  padding = 'same')
        self.reluOut2 = nn.ReLU() 
        
    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): [batch_size, in_channels, n_mels, n_frames]
        Returns:
            out (torch.Tensor): [batch_size, 1, n_mels, n_frames]       
            
        """
        x = self.convOut1(x)
        x = self.reluOut1(x)
        x = self.convOut2(x)
        x = self.reluOut2(x)

        # Try to normalize each batch independently
        out = x / torch.max(x)
        
        return out

class PInvBlock(nn.Module):
    def __init__(self, hparams):

        super(PInvBlock, self).__init__()
        self.melfb = torch.as_tensor(librosa.filters.mel(sr = hparams.sr, 
                                                         n_fft = hparams.n_fft, 
                                                         n_mels = hparams.n_mels)).to(config.DEVICE)
        
    
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]

        Returns:
            _type_: _description_
        """
        stft_hat = torch.clamp(torch.matmul(torch.linalg.pinv(self.melfb), melspec), min=0, max=1)
        return stft_hat