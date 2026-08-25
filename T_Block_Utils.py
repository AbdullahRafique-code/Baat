# defining LayerNorm, GELU and Feedforward MLP

import torch
import torch.nn as nn
import model_config

class LayerNorm(nn.Module):
    def __init__(self,config:model_config):
        super().__init__()
        self.eps=1e-5 # to precvent division by zero
        self.scale=nn.Parameter(torch.ones(config.embedding_dim)) # learnable scale parameter
        self.shift=nn.Parameter(torch.zeros(config.embedding_dim)) #learnable shift parameter

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keedim=True,unbiased=False)
        normalized_x=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*normalized_x+self.shift


# now the GELU activation function (smooth version of ReLU)
class GELU(nn.Module):
    

                          
                                  