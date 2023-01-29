import torch
import torch.nn as nn

class ConvGLUBlock(nn.Module):  
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 last_block = False):
        
        super(ConvGLUBlock, self).__init__()
        self.last_block = last_block
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              padding = "same")
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.glu = nn.GLU(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.conv(x)
        if not self.last_block:
            x = self.bn(x)
            x = self.glu(x)
            
        return x
    
class AIGC(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size):
        
        super(AIGC, self).__init__()
        self.conv_re = nn.Conv2d(in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 bias = False,
                                 padding="same")
        self.conv_im = nn.Conv2d(in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 bias = False,
                                 padding="same")
        self.conv_gate = nn.Conv2d(in_channels = in_channels+1,
                                   out_channels = out_channels,
                                   kernel_size = kernel_size,
                                   bias = False,
                                   padding="same")
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_stft_hat, x_stft_mag):
        
        x_stft_hat_real = x_stft_hat.real.float()
        x_stft_hat_imag = x_stft_hat.imag.float()
        conv_real = self.conv_re(x_stft_hat_real) - self.conv_im(x_stft_hat_imag)
        conv_imag = self.conv_re(x_stft_hat_imag) + self.conv_im(x_stft_hat_real)
        ampgate = self.sigmoid(self.conv_gate(torch.cat([torch.abs(x_stft_hat).float(), x_stft_mag.unsqueeze(1).float()], axis=1)))

        out = conv_real * ampgate + 1j * conv_imag * ampgate
        
        return out
    
class LastConv(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size = (1,1)):
        
        super(LastConv, self).__init__()
        self.conv_re = nn.Conv2d(in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 bias = False,
                                 padding="same")
        self.conv_im = nn.Conv2d(in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 bias = False,
                                 padding="same")
        
    def forward(self, x_stft_hat):
        
        x_stft_hat_real = x_stft_hat.real
        x_stft_hat_imag = x_stft_hat.imag     
        
        conv_real = self.conv_re(x_stft_hat_real) - self.conv_im(x_stft_hat_imag)
        conv_imag = self.conv_re(x_stft_hat_imag) + self.conv_im(x_stft_hat_real)
        
        out = conv_real + 1j * conv_imag
        
        return out
        