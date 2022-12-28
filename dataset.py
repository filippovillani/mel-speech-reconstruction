import torch
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import os

from utils.audioutils import to_db, normalize_db_spectr


def build_dataloader(hparams: Namespace,
                     data_dir: str,
                     task: str,
                     set_type: str = "train") -> DataLoader:
    
    shuffle = True if set_type == "train" else False
    
    if task == "stft2wav":
        ds = STFTDataset(data_dir, 
                         set_type = set_type)
    elif task == "mel2stft":
        ds = SpectrogramDataset(data_dir, 
                                set_type = set_type)
    else:
        raise ValueError(f"task must be one of [stft2wav, mel2stft], received {task}")
    
    dataloader = DataLoader(ds, 
                            batch_size = hparams.batch_size,
                            num_workers = hparams.num_workers, 
                            shuffle=shuffle)
    return dataloader
    
class SpectrogramDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 set_type: str = "train"):
        super(SpectrogramDataset, self).__init__()
        self.data_dir = data_dir
        self.spectr_dir = data_dir / "stft" / set_type
        self.spectr_list_path = [self.spectr_dir / path for path in os.listdir(self.spectr_dir)]
       
    def __getitem__(self, idx):
        
        spectr = normalize_db_spectr(to_db(torch.abs(torch.load(self.spectr_list_path[idx]))))
        return {'spectrogram': spectr}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))
    
class STFTDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 set_type: str = "train"):
        
        super(STFTDataset, self).__init__()
        self.data_dir = data_dir
        self.spectr_dir = data_dir / "stft" / set_type
        self.spectr_list_path = [self.spectr_dir / path for path in os.listdir(self.spectr_dir)]
       
    def __getitem__(self, idx):
        
        stft = torch.load(self.spectr_list_path[idx])
        return {'stft': stft}
        
    def __len__(self):
        return len(os.listdir(self.spectr_dir))


