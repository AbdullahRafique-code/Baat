import torch.nn as nn
import model_config
from MultiAttentionHead import MultiAttentionHead


class TransformerBlock(nn.Module):
    def __init__(self, config:model_config):
        super().__init__()

        #layerNorm
        self.ln1=nn.LayerNorm(config.dim) #768

        # using our own custom MultiAttentionHead
        self.attn=MultiAttentionHead(dim_in=config.dim,dim_out=config.dim,
                                    context_length=config.context_length,dropout=config.dropout,
                                    num_heads=config.num_heads)

        #LayerNorm again
        self.ln2=nn.LayerNorm(config.dim)

        #Feedforward MLP
        self.mlp=nn.Sequential(
            nn.Linear(config.dim,4*config.dim),
            nn.GELU(),
            nn.Linear(4*config.dim,config.dim)
        )


    def forward(self,x):
        # masking the future tokens, using a mask
        seq_len=x.size(1)
        casual_mask=nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # norm1
        norm_x=self.ln1(x)

        attn_out=self.attn(norm_x,norm_x,norm_x,is_causal=True,attn_mask=casual_mask)[0]
        #shortcut
        x=x+attn_out

        #feedforward
        x=x+self.mlp(self.ln2(x))
        return x