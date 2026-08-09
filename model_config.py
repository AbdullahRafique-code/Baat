from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaatConfig:
    vocab_size =32000
    dim=  768
    num_layers=12
    num_heads=12
    max_seq_len=1024
    dropout=0.1
