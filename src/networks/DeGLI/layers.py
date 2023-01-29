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