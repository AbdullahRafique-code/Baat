import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
import model_config


# Defining the main architecture

class BaatLLM(nn.Module):
    def __init__(self, config:model_config):
        super().__init__()
        self.config=config

        # tokens to embedding vectors + pos embeddings + dropout
        self.tok_emb=nn.Embedding(config.vocab_size,config.dim)
        self.pos_emb=nn.Embedding(config.max_seq_len,config.dim)
        self.drop=nn.Dropout(config.dropout)

        # The 12 block stack
        self.blocks=nn.Sequential(*[TrasnformerBlock(config) for _ in range(config.num_layers)])

        #Final Output Head
        self.lnf=nn.LayerNorm(config.dim)
        self.out=nn.Linear(config.dim,config.vocab_size,bias=False)
        self.out.weight=self.tok_emb.weight

    def forward(self,tokens):
        batch_size,seq_len=tokens.shape

        #adding emb vector + pos emb
        positions=torch.arange(0,seq_len,dtype=torch.long ,device=tokens.device)
        x=self.tok_emb(tokens)+self.pos_emb(positions)
        x=self.drop(x)

        # the 12 blocks
        x=self.blocks(x)

        # final output 
        
        x=self.lnf(x)
        x=self.out(x)
        return x


        