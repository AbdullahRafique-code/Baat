# defining the pretraining code and evaluation
import os
import math # cosine decay
import time # time/sped calc
import torch
import torch.nn.functional as f # cross entropy loss

from model_config import BaatConfig
from modelARC import BaatLLM
from dataloaders import get_dataloader


#Parameters 
 
warmup_steps=2000 # slowly increasing to LR prevent rndm weigt breakingm (as random at start)
max_lr=5e-4
min_lr=5e-4
batch_size=32
context_length=1024
Grad_cl=1.0 # to prevent mathemtiacal exp, gradient vanisihng
checkpoint_dir="checkpoints"
max_steps=2_000_000_000/(batch_size*context_length) # 2B tok (/32 batchsize*1024 context length)
#61000

#cosaine decay and LR
def get_lr(step):
    #linearing going up
    if step<warmup_steps:
        return max_lr*(step/warmup_steps)

    #stop point
    if step>max_steps:
        return min_lr

    #Cosine decay
    decay_ratio=(step-warmup_steps)/(max_steps-warmup_steps)
    coefficient=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coefficient*(max_lr-min_lr)

