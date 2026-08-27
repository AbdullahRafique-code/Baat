from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaatConfig:
    vocab_size:int =32000
    dim:int = 256 # nano model test  #1024 # increased from 768
    num_layers:int = 4# nano model test #14 # prev 12
    num_heads:int =4 # nano model test  #16 # prev 12 (1024/16=64 dim per head)
    context_length:int =128 # nano model test  # 1024 context length
    dropout:float =0.1

# size calc
# emb = vocab_size * dim = 32,000 * 1024 = 32,768,000
# positional emb = context_length * dim = 1024 * 1024 = 1,048,576
# Blocks paramaters=14*( Attn (4.19M) +MLP (8.38M) +LN )= 14*$12,587,008=176,218,112
# final layernorm= dim*2 = 1024*2 = 2,048
# Total parameters = 32,768,000 + 1,048,576 + 176,218,112 + 2,048 = 210,036,736