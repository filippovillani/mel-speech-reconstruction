import torch
import torch.nn as nn
from argparse import Namespace
import librosa 

import config

# mel -> stft ->  wav

# mel -> stft: enc/dec structure?
# stft -> wav: Bi-LSTM on time-samples to correct the prediction

class MelSpec2Spec(nn.Module):
    def __init__(self,
                 hparams: Namespace):
        super().__init__()
        self.hprms = hparams
        self.melfb = torch.as_tensor(librosa.filters.mel(sr=self.hprms.sr, 
                                                    n_fft=self.hprms.n_fft, 
                                                    n_mels = self.hprms.n_mels)).to(config.DEVICE)
        
        self.conv_block0 = self._conv2d_block(hparams.in_channels[0],
                                              hparams.out_channels[0],
                                              hparams.kernel_size,
                                              upsample=False)
        self.maxpool1 = nn.MaxPool2d((3,1))
        
        num_layers = len(hparams.in_channels)
        self.conv_blocks = nn.Sequential(*[self._conv2d_block(hparams.in_channels[l], 
                                                              hparams.out_channels[l],
                                                              hparams.kernel_size) for l in range(1, num_layers)])
    
        # self.convOut1 = nn.Conv2d(64, 2, 3, padding='same')
        # self.reluOut1 = nn.ReLU() 
        self.out = nn.Conv2d(hparams.out_channels[-1], 1, (4,3), padding=(2,1))
        self.sigmoid = nn.Sigmoid()
    
    def _compute_mel_spectrogram(self, x):
        out = torch.empty((self.hprms.batch_size,
                           self.hprms.n_channels,
                           self.hprms.n_mels, 
                           self.hprms.n_frames))  
        
        for n in range(x.shape[0]):
            inp = x[n].squeeze()
            out[n] = torch.matmul(self.melfb, inp).unsqueeze(0)
        
        return out
        
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
        melspec = melspec.unsqueeze(1)
        x = self.conv_block0(melspec)
        x = self.maxpool1(x)
        x = self.conv_blocks(x)
        x = self.out(x)
        
        return self._compute_mel_spectrogram(x)
        
        
if __name__=="__main__":
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_channels, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = MelSpec2Spec(hparams)
    print(model(batch).shape)
    
