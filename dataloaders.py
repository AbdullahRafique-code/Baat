import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

class BaatDataset(Dataset):
    def __init__(self,bin_path,context_length=1024,start_sample=0):
        self.context_length=context_length
        # memory mapping the binary file, read only and luint16
        self.data=np.memmap(bin_path,dtype=np.uint16,mode='r')

        total_samples = (len(self.data) - 1) // self.context_length
        self.start_sample = start_sample
        self.num_samples = max(0, total_samples - start_sample)

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):

        actual_idx = self.start_sample + idx
        start=idx *self.context_length
        end=start+self.context_length+1
    
        chunk=torch.from_numpy(self.data[start:end].astype(np.int64))

        x=chunk[:-1]
        y=chunk[1:]
        return x,y



def get_dataloader(bin_path,batch_size=32,context_length=1024,num_workers=4,start_sample=0):
    dataset=BaatDataset(bin_path,context_length=context_length,start_sample=start_sample)

    loader= DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,
    pin_memory=True,drop_last=True)
    return loader




