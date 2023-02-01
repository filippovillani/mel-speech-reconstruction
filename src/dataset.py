import os
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(hparams: Namespace,
                     data_dir: str,
                     ds_type: str = "train") -> DataLoader:
    
    shuffle = True if ds_type == "train" else False
 
    ds = STFTDataset(data_dir, 
                     ds_type = ds_type)
    
    dataloader = DataLoader(ds, 
                            batch_size = hparams.batch_size,
                            num_workers = hparams.num_workers, 
                            shuffle = shuffle,
                            drop_last = True)
    return dataloader
    
    
class STFTDataset(Dataset):
    def __init__(self, 
                 data_dir: Path,
                 ds_type: str = "train"):
        
        super(STFTDataset, self).__init__()
        self.spectr_dir = data_dir / "stft" / ds_type
        self.spectr_list_path = [self.spectr_dir / path for path in os.listdir(self.spectr_dir)]
       
    def __getitem__(self, idx):
        
        stft = torch.load(self.spectr_list_path[idx])
        return {'stft': stft}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))


