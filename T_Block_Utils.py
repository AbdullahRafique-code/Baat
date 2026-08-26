# defining LayerNorm, GELU and Feedforward MLP

import torch
import torch.nn as nn
import model_config

class LayerNorm(nn.Module):
    def __init__(self,config:model_config.BaatConfig):
        super().__init__()
        self.eps=1e-5 # to precvent division by zero
        self.scale=nn.Parameter(torch.ones(config.dim)) # learnable scale parameter
        self.shift=nn.Parameter(torch.zeros(config.dim)) #learnable shift parameter

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        normalized_x=(x-mean)/torch.sqrt(var+self.eps)
        return self.scale*normalized_x+self.shift


# now the GELU activation function (smooth version of ReLU)
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))
                                        *(x+0.044715*torch.pow(x,3))))

# the MLP feedforward

class FeedForward(nn.Module):
    def __init__(self,config:model_config.BaatConfig):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(config.dim,4*config.dim),
                                  GELU(),
                                  nn.Linear(4*config.dim,config.dim))

    def forward(self,x):
        return self.layers(x)