import torch
from torch.utils.data import Dataset, DataLoader, random_split
from argparse import Namespace
import numpy as np
import os
from typing import Tuple

import config

def build_dataloaders(data_dir: str,
                      hparams: Namespace)->Tuple[DataLoader, DataLoader]:
    ds = MelSTFTDataset(data_dir)
    train_ds, val_ds = random_split(ds, [0.7, 0.3], generator=torch.Generator().manual_seed(config.SEED))
    train_dl = DataLoader(train_ds, 
                          hparams.batch_size, 
                          shuffle=True, 
                          pin_memory=True)
    val_dl = DataLoader(val_ds, 
                        hparams.batch_size, 
                        shuffle=False,
                        pin_memory=True)
    
    return train_dl, val_dl
    
class MelSTFTDataset(Dataset):
    def __init__(self, 
                 data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.spectr_dir = data_dir / "spectr" 
        self.melspectr_dir = data_dir / "melspectr" 
        
    def __getitem__(self, 
                    idx):
        
        melspectr = torch.from_numpy(np.load(os.listdir(self.melspectr_dir)[idx]))
        spectr = torch.from_numpy(np.load(os.listdir(self.spectr_dir)[idx]))
        
        return {'melspectr': melspectr,
                'spectr': spectr}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))
