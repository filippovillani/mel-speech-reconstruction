import os
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from utils.audioutils import to_db, normalize_db_spectr

def build_dataloader(hparams: Namespace,
                     data_dir: str,
                     task: str = "melspec2spec",
                     ds_type: str = "train") -> DataLoader:
    
    shuffle = True if ds_type == "train" else False
    
    if task in ["melspec2spec", "melspec2wav"]:
        ds = SpectrogramDataset(data_dir, 
                                ds_type = ds_type)
    elif task == ["spec2wav"]:
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

class SpectrogramDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 ds_type: str = "train"):
        super(SpectrogramDataset, self).__init__()
        self.data_dir = data_dir
        
        self.spectr_dir = data_dir / "stft" / ds_type
        
        self.spectr_list_path = [self.spectr_dir / path for path in os.listdir(self.spectr_dir)]
       
    def __getitem__(self, idx):
        
        spectr = normalize_db_spectr(to_db(torch.abs(torch.load(self.spectr_list_path[idx]))))
        
        return {'spectrogram': spectr}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))
