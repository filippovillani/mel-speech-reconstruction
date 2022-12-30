import torch
import torch.nn as nn
    
class ConvDNN(nn.Module):  
    def __init__(self, hparams):
        
        super(ConvDNN, self).__init__()
        self.convblock1 = ConvBlock(in_channels = 6,
                                    out_channels = hparams.hidden_channel,
                                    kernel_size = hparams.kernel_size)
        self.convblock23 = nn.Sequential(ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = hparams.kernel_size),
                                         ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = hparams.kernel_size))
        self.convblock45 = nn.Sequential(ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = hparams.hidden_channel,
                                                   kernel_size = hparams.kernel_size),  
                                         ConvBlock(in_channels = hparams.hidden_channel // 2,
                                                   out_channels = 2,
                                                   kernel_size = hparams.kernel_size,
                                                   last_block = True))
        
    def forward(self, x, x_amp_proj, x_cons_proj):
        
        x = torch.cat([x, x_amp_proj, x_cons_proj], axis=1) 
        x = self.convblock1(x)
        x = self.convblock23(x) + x
        x = self.convblock45(x)
        
        return x
            
class ConvBlock(nn.Module):  
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 last_block = False):
        
        super(ConvBlock, self).__init__()
        self.last_block = last_block
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = "same")
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.glu = nn.GLU(dim=1)
        
    def forward(self, x):
        
        x = self.conv(x)
        if not self.last_block:
            x = self.bn(x)
            x = self.glu(x)
        
        return x