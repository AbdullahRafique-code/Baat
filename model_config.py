from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaatConfig:
    vocab_size:int =32000
    dim:int = 768
    num_layers:int =12
    num_heads:int =12
    context_length:int =1024 #context length
    dropout:float =0.1
