import torch
from torch.utils.data import Dataset, DataLoader, random_split
from argparse import Namespace
import numpy as np
import os
from typing import Tuple

from audioutils import to_db, normalize_db_spectr
import config

def build_dataloader(hparams: Namespace,
                     data_dir: str,
                     type: str = "train") -> DataLoader:
    
    shuffle = True if type == "train" else False
    
    ds = SpectrogramDataset(data_dir, 
                        type = type)
    dataloader = DataLoader(ds, 
                            hparams.batch_size,
                            num_workers = hparams.num_workers, 
                            shuffle=shuffle)

    
    return dataloader
    
class SpectrogramDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 type: str = "train"):
        super(SpectrogramDataset, self).__init__()
        self.data_dir = data_dir
        
        self.spectr_dir = data_dir / "spectrograms" / type
        
        self.spectr_list_path = [self.spectr_dir / path for path in os.listdir(self.spectr_dir)]
       
    def __getitem__(self, idx):
        
        spectr = normalize_db_spectr(to_db(torch.load(self.spectr_list_path[idx])))
        
        return {'spectrogram': spectr}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))
    
train_dl = build_dataloader(config.create_hparams(), config.DATA_DIR, "train")
val_dl = build_dataloader(config.create_hparams(), config.DATA_DIR, "validation")
