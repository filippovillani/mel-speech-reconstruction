import torch
import torch.nn as nn
from argparse import Namespace

import config

# mel -> stft ->  wav

# mel -> stft: enc/dec structure?
# stft -> wav: Bi-LSTM on time-samples to correct the prediction

class Network(nn.Module):
    def __init__(self,
                 hparams: Namespace):
        super().__init__()
        num_layers = len(hparams.in_channels)
        self.conv_block0 = self._conv2d_block(hparams.in_channels[0],
                                              hparams.out_channels[0],
                                              hparams.kernel_size,
                                              upsample=False)
        self.maxpool1 = nn.MaxPool2d((3,1))
        
        self.conv_blocks = nn.Sequential(*[self._conv2d_block(hparams.in_channels[l], 
                                                              hparams.out_channels[l],
                                                              hparams.kernel_size) for l in range(1, num_layers)])
    
        self.out = nn.Conv2d(hparams.out_channels[-1], 1, (4,3), padding=(2,1))
        

    def _conv2d_block(self,
                      in_channels,
                      out_channels,
                      kernel_size,
                      upsample=True):
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        bn = nn.BatchNorm2d(out_channels)
        act = nn.ReLU()
        
        if upsample:
            upsamp = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
            block = nn.Sequential(conv, bn, act, upsamp)
        else:
            block = nn.Sequential(conv, bn, act)
        
        return block
    
    def forward(self, melspec):
        print(melspec.shape)
        x = self.conv_block0(melspec)
        x = self.maxpool1(x)
        print(x.shape)
        x = self.conv_blocks(x)
        print(x.shape)
        stft = self.out(x)
        return stft
        
        
if __name__=="__main__":
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_channels, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = Network(hparams)
    print(model(batch).shape)
    
