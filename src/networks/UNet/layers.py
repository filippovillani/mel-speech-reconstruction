import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 drop_rate):
        
        super(Bottleneck, self).__init__()
        
        out_channels = in_channels * 2 
        self.conv1 = nn.Conv2d(in_channels = in_channels, 
                                out_channels = out_channels, 
                                kernel_size = kernel_size, 
                                padding = 'same')
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(out_channels)        
        self.relu1 = nn.ReLU() 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU() 
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)        
        x = self.relu1(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop(x)
        return x     
        
class ContractingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 drop_rate,
                 out_channels = None):

        super(ContractingBlock, self).__init__()
        
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
        self.dropC = nn.Dropout(drop_rate)
        self.poolC = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): [batch_size, in_channels, n_mels, n_frames]

        Returns:
            out (torch.Tensor):
                if in_channels == 1: in_channels = hparams.first_unet_channel_units
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
        x = self.poolC(x_cat)
        x = self.dropC(x)
        return x, x_cat
    

class ExpandingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 drop_rate):

        super(ExpandingBlock, self).__init__()
        
        out_channels = in_channels // 2

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
        self.dropE = nn.Dropout(drop_rate)
        
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
        if x.shape != x_cat.shape:
            x = self._reshape_x_for_cat(x, x_cat)
        x = torch.cat((x, x_cat), axis=1) 

        
        x = self.convE1(x) 
        x = self.bnE1(x) 
        x = self.reluE1(x)
        x = self.convE2(x)
        x = self.bnE2(x)
        x = self.reluE2(x)
        x = self.dropE(x)
        
        return x
    
    def _reshape_x_for_cat(self, x, x_cat):
        x_shape = x.shape
        x_cat_shape = x_cat.shape
        
        # Find the difference in shape between x and x_cat
        pad_sizes = [x_cat_shape[i] - x_shape[i] for i in range(len(x_shape))]
        
        # Add zeros to the last dimension of x
        pad = []
        for n in range(len(pad_sizes)-1,-1,-1):
            pad.extend([0, pad_sizes[n]])
        pad = tuple(pad)
        x = F.pad(x, pad=pad, mode='constant', value=0)

        return x

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

        return x