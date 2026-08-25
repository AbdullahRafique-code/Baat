import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import model_config

class BaatDataset(Dataset):
    def __init__(self,bin_path,seq_len=1024):
        self.seq_len=seq_len
        # memory mapping the binary file, read only and luint16
        self.data=np.memmap(bin_path,dtype=np.uint16,mode='r')

        self.num_samples=(len(self.data)-1)//self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        start=idx *self.seq_len
        end=start+self.seq_len+1
    
        chunk=torch.from_numpy(self.data[start:end].astype(np.int64))

        x=chunk[:-1]
        y=chunk[1:]
        return x,y


def get_dataloader(bin_path,batch_size=32,seq_len=1024,num_workers=4):
    dataset=BaatDataset(bin_path,seq_len=seq_len)

    loader= DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,
    pin_memory=True,drop_last=True)
    return loader




