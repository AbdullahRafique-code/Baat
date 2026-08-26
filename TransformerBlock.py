import torch.nn as nn
import model_config
from MultiAttentionHead import MultiAttentionHead
from T_Block_Utils import LayerNorm,GELU,FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config:model_config.BaatConfig):
        super().__init__()

        #layerNorm
        self.ln1=LayerNorm(config.dim) #768

        # using our own custom MultiAttentionHead
        self.attn=MultiAttentionHead(dim_in=config.dim,dim_out=config.dim,
                                    context_length=config.context_length,dropout=config.dropout,
                                    num_heads=config.num_heads)

        #LayerNorm again
        self.ln2=LayerNorm(config.dim)

        #feedforward MLP
        self.mlp=FeedForward(config)

        #shortcut drop
        self.drop_shortcut=nn.Dropout(config.dropout)

    def forward(self,x):
        #attention
        shortcut=x
        x=self.ln1(x)
        x=self.attn(x)

        x=self.drop_shortcut(x)
        x=x+shortcut

        #feedforward
        shortcut=x
        x=self.ln2(x)
        x=self.mlp(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
        return x