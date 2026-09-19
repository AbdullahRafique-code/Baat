from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class BaatConfig:
    vocab_size:int =32000
    dim:int = 768 # lowering down again # 256 # nano model test  #1024 # increased from 768
    num_layers:int = 12 # lowering down again # 4# nano model test #14 # prev 12
    num_heads:int =12 # lowering down again #  4 # nano model test  #16 # prev 12 (1024/16=64 dim per head)
    context_length:int =1024#128 # nano model test  # 1024 context length
    dropout:float =0.1

# size calc
# emb = vocab_size * dim = 32,000 * 768 = 24,576,000
# positional emb = context_length * dim = 1024 * 768 = 786,432
# Blocks paramaters=12*( Attn  +MLP  +LN ) =85,054,464
# final layernorm= dim*2 = 768*2 = 1,536
# Total parameters = 24,576,000 + 786,432 + 1,536 + 85,054,464 = 109,408,064 ~ 109M parameters