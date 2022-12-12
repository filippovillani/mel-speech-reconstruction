import torch
import torch.nn as nn
from argparse import Namespace
import librosa 

import config

        
class MelSpect2Spec(nn.Module):
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
                                              hparams.dropout,
                                              upsample=False)
        
        num_layers = len(hparams.in_channels)
        self.conv_blocks = nn.Sequential(*[self._conv2d_block(hparams.in_channels[l], 
                                                              hparams.out_channels[l],
                                                              hparams.kernel_size,
                                                              hparams.dropout) for l in range(1, num_layers)])
    

        self.out = nn.Conv2d(hparams.out_channels[-1], 1, (4,3), padding=(2,1))
        self.out_act = nn.ReLU()
            
    def _conv2d_block(self,
                      in_channels,
                      out_channels,
                      kernel_size,
                      dropout,
                      upsample=True):
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        bn = nn.BatchNorm2d(out_channels)
        act = nn.ReLU()
        drop = nn.Dropout(dropout)
        
        if upsample:
            upsamp = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
            block = nn.Sequential(conv, bn, act, drop, upsamp)
        else:
            pool = self.maxpool0 = nn.MaxPool2d((3,1))
            block = nn.Sequential(conv, bn, act, drop, pool)
        
        return block
    
    def compute_mel_spectrogram(self, stft_spectrogram):
        if stft_spectrogram.dim() == 2:
            out = torch.matmul(self.melfb, stft_spectrogram)
        else:
            stft_spectrogram = stft_spectrogram.squeeze()
            out = torch.empty((stft_spectrogram.shape[0],
                            self.hprms.n_mels,
                            self.hprms.n_frames)).to(config.DEVICE) 
            
            for n in range(out.shape[0]):
                out[n] = torch.matmul(self.melfb, stft_spectrogram[n])
        
        return out
    
    def compute_stft_spectrogram(self, mel_spectrogram):
        melspec = mel_spectrogram.unsqueeze(1)
        x = self.conv_block0(melspec)
        x = self.conv_blocks(x)
        stftspec_hat = self.out(x)
        
        return self.out_act(stftspec_hat)
    
    def forward(self, melspec):
        stftspec_hat = self.compute_stft_spectrogram(melspec)
        melspec_hat = self.compute_mel_spectrogram(stftspec_hat)
        
        return melspec_hat
        
        
if __name__=="__main__":
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = MelSpect2Spec(hparams).to(config.DEVICE)
    print(model(batch).shape)
    print(model.compute_stft_spectrogram(batch).squeeze().shape)
    
