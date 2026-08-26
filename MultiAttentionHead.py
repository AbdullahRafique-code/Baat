# Chosing to make own version of the Multi Attention Head
import torch
import torch.nn as nn

class MultiAttentionHead(nn.Module):
    def __init__(self, dim_in, dim_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert(dim_out%num_heads==0),"dim_out must be divisible by num_heads"
        self.dim_out=dim_out
        self.num_heads=num_heads
        self.head_dim=dim_out//num_heads
        self.W_query=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_key=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_value=nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.out_proj=nn.Linear(dim_out,dim_out)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer("mask",torch.tril(torch.ones(context_length,context_length)),diagonal=1)

    def forward(self,x):
        b,num_tokens,dim_in=x.shape
        keys=self.W_key(x)
        queries=self.W_query(x)
        values=self.W_value(x)

        # reshape for multihead attention
        keys=keys.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        queries=queries.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        values=values.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)

        #calculate attention scores
        attn_scores=queries@keys.transpose(2,3)

        mask_bool=self.mask.bool()[:num_tokens,:num_tokens] # mask turncates to num tokens at current elvel

        atten_scores_masked=attn_scores.masked_fill(mask_bool,-torch.inf)

        atten_weights=torch.softmax(atten_scores_masked/keys.shape[-1]**0.5,dim=-1)

        atten_weights=self.dropout(atten_weights)

        #context vector
        context_vector=(atten_weights@values).transpose(1,2)

        context_vector=context_vector.contiguous().view(b,num_tokens,self.dim_out)

        # final output projection
        context_vector=self.out_proj(context_vector)
        return context_vector