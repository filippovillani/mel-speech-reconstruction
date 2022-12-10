import torch
import torch.nn as nn
from argparse import Namespace
import librosa 

from griffinlim import fast_griffin_lim
import config

# mel -> stft ->  wav

# mel -> stft: enc/dec structure?
# stft -> wav: Bi-LSTM on time-samples to correct the prediction

class MelSpec2Wav(nn.Module):
    def __init__(self, 
                 hparams: Namespace):
        self.hprms = hparams
        self.melspec2spec = MelSpec2Spec(hparams)
    
    def _compute_gla(self, spectr):
        wav = fast_griffin_lim(spectr,
                               n_fft = self.hprms.n_fft,
                               sr = self.hprms.sr,
                               num_iter=100)
        return wav       
    
    def forward(self, melspec):
        spec = self.melspec2spec(melspec)
        wav = self._compute_gla(spec) # TODO: compute_gla to torch (now np), try to change GLA with NNs
        return wav

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
        self.maxpool0 = nn.MaxPool2d((3,1))
        
        num_layers = len(hparams.in_channels)
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
    
    def compute_mel_spectrogram(self, spectr):
        spectr = spectr.squeeze()
        out = torch.empty((spectr.shape[0],
                           self.hprms.n_mels,
                           self.hprms.n_frames)).to(config.DEVICE) 
        
        for n in range(out.shape[0]):
            inp = spectr[n]
            out[n] = torch.matmul(self.melfb, inp)
        
        return out
    
    def forward(self, melspec):
        melspec = melspec.unsqueeze(1)
        x = self.conv_block0(melspec)
        x = self.maxpool0(x)
        x = self.conv_blocks(x)
        spectr_hat = self.out(x) 
        
        return spectr_hat
        
        
if __name__=="__main__":
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = MelSpec2Spec(hparams).to(config.DEVICE)
    print(model(batch).shape)
    
