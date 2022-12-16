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

        self.nn = nn.Sequential(nn.Conv2d(1, 8, 3, padding='same'),
                                 nn.BatchNorm2d(8),
                                 nn.ReLU(),
                                #  nn.MaxPool2d(2),
                                 nn.Dropout(0.3),
                                 nn.Conv2d(8, 16, 3, padding='same'),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Conv2d(16, 32, 3, padding='same'),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Conv2d(32, 16, 3, padding='same'),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Conv2d(16, 1, 3, padding='same'),
                                 nn.ReLU())


    # def compute_mel_spectrogram(self, stft_spectrogram):
    #     if stft_spectrogram.dim() == 2:
    #         melspec = torch.matmul(self.melfb, stft_spectrogram)
    #     else:
    #         stft_spectrogram = stft_spectrogram.squeeze()
    #         melspec = torch.empty((stft_spectrogram.shape[0],
    #                         self.hprms.n_mels,
    #                         self.hprms.n_frames)).to(config.DEVICE) 
            
    #         for n in range(melspec.shape[0]):
    #             melspec[n] = torch.matmul(self.melfb, stft_spectrogram[n])
        
    #     return melspec
    
    def forward(self, melspec):
        stft_hat = torch.matmul(torch.linalg.pinv(self.melfb), melspec)
        stft_hat = self.nn(stft_hat)
        
        return stft_hat
        
        
if __name__=="__main__":
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_channels, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = MelSpect2Spec(hparams).to(config.DEVICE)
    print(batch.shape)
    print(model(batch).shape)
    
