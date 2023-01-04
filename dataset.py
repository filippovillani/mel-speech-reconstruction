import os
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(hparams: Namespace,
                     data_dir: str,
                     set_type: str = "train") -> DataLoader:
    
    shuffle = True if set_type == "train" else False
 
    ds = STFTDataset(data_dir, 
                     set_type = set_type)
    
    dataloader = DataLoader(ds, 
                            batch_size = hparams.batch_size,
                            num_workers = hparams.num_workers, 
                            shuffle = shuffle,
                            drop_last = True)
    return dataloader
    
    
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


