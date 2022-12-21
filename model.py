import torch
import torch.nn as nn
from torchsummary import summary
import librosa 

from layers import ContractingBlock, ExpandingBlock, PInvBlock, OutBlock
import config

def build_model(hparams,
                model_name,
                weights_dir = None,
                best_weights: bool = True):
    
    if model_name == "unet":
        model = UNet(hparams).float().to(config.DEVICE)
    elif model_name == "convpinv":
        model = ConvPInv(hparams).float().to(config.DEVICE)
    
    if weights_dir is not None:
        weights_path = 'best_weights' if best_weights else 'ckpt_weights'
        weights_path = config.WEIGHTS_DIR / weights_dir / weights_path
        model.load_state_dict(torch.load(weights_path))
    
    return model 

class ConvPInv(nn.Module):
    def __init__(self, hparams):

        super(ConvPInv, self).__init__()
        
        self.pinvblock = PInvBlock(hparams)
        self.conv1 = nn.Conv2d(in_channels = hparams.n_channels,
                               out_channels = hparams.first_channel_units,
                               kernel_size = hparams.kernel_size,
                               padding = 'same')
        self.bn1 = nn.BatchNorm2d(hparams.first_channel_units)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = hparams.first_channel_units,
                               out_channels = hparams.n_channels,
                               kernel_size = hparams.kernel_size,
                               padding = 'same')
        self.bn2 = nn.BatchNorm2d(hparams.n_channels)
        self.relu2 = nn.ReLU()
        
    
    def forward(self, melspec):
        """
        Args:
            melspec (torch.Tensor): mel spectrogram in dB normalized in [0, 1]

        Returns:
            _type_: _description_
        """
        x = self.pinvblock(melspec)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        stft_hat = x / torch.max(x)
        return stft_hat

class UNet(nn.Module):
    def __init__(self, hparams):
        
        super(UNet, self).__init__()
        
        self.pinvblock = PInvBlock(hparams)
        self.contrblock1 = ContractingBlock(in_channels = hparams.n_channels,
                                            out_channels = hparams.first_channel_units,
                                            kernel_size = hparams.kernel_size)
        self.contrblock2 = ContractingBlock(in_channels = hparams.first_channel_units,
                                            kernel_size = hparams.kernel_size)

        self.contrblock3 = ContractingBlock(in_channels = hparams.first_channel_units * 2,
                                            kernel_size = hparams.kernel_size,
                                            last_block = True)

        self.expandblock2 = ExpandingBlock(in_channels = hparams.first_channel_units * 4,
                                           kernel_size = hparams.kernel_size)
        self.expandblock1 = ExpandingBlock(in_channels = hparams.first_channel_units * 2,
                                           kernel_size = hparams.kernel_size,
                                           last_block = True)
        self.outblock = OutBlock(in_channels = hparams.first_channel_units)
        
    def forward(self, melspec):
        stft_hat = self.pinvblock(melspec)
        x, x_cat1 = self.contrblock1(stft_hat)
        x, x_cat2 = self.contrblock2(x)
        x, _ = self.contrblock3(x)
        x = self.expandblock2(x, x_cat2)
        x = self.expandblock1(x, x_cat1)
        out = self.outblock(x)
        
        return out
     
 
# Debug model:
if __name__ == "__main__":
    
    hparams = config.create_hparams()
    batch = torch.rand((hparams.batch_size, hparams.n_channels, hparams.n_mels, hparams.n_frames)).to(config.DEVICE)
    
    model = ConvPInv(hparams).to(config.DEVICE)
    
    print(batch.shape)
    print(model(batch).shape)
    print(summary(model, batch))

    
