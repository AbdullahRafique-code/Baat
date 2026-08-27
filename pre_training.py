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
 
warmup_steps= 10 # nano model test #2000 # slowly increasing to LR prevent rndm weigt breakingm (as random at start)
max_lr=5e-4
min_lr=5e-5
batch_size=2 # 4 for testing on my machine : 32 for training actually 
context_length=128 # nano model test
Grad_cl=1.0 # to prevent mathemtiacal exp, gradient vanisihng
checkpoint_dir="checkpoints"
max_steps=50 # nano model test  #4_200_000_000//(batch_size*context_length) # 4B tok (/32 batchsize*1024 context length)
#Roughly 128173

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


# val function
@torch.no_grad()
def evaluate(model, val_loader, device,ptdtype,eval_iters=100):
 model.eval() # set to eval mode
 losses=torch.zeros(eval_iters)
 for k,(x,y) in enumerate(val_loader):
     if k>=eval_iters:
            break
     x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
     with torch.autocast(device_type=device.type,dtype=ptdtype):
            logits=model(x)
            loss=f.cross_entropy(logits.view(-1,model.config.vocab_size),y.view(-1))
     losses[k]=loss.item()

 model.train() # set back to train mode
 return losses.mean().item() # return mean loss over eval_iters





# Hardware check and device selection
def train():
    device=torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    

    # checker for fp16 or bfloat16
    cuda_cap=torch.cuda.get_device_capability() if torch.cuda.is_available() else(0,0)
    if cuda_cap[0]>=8:
        print("Using bfloat16 precision for training. No scalar needed")
        floattype=torch.bfloat16
        use_scaler=False
    else:
        print("Using float16 precision for training. Using scalar for gradient scaling")
        floattype=torch.float16
        use_scaler=True


    os.makedirs(checkpoint_dir,exist_ok=True)
    config=BaatConfig()
    model=BaatLLM(config).to(device) # load model, pass config,move to device

    # speeding up model with torch.compile (if available)
    if cuda_cap[0]>=7:
        try:
            model=torch.compile(model) # compile model for speedup
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    else:
        print("Model compilation not supported on this device. Proceeding without compilation.")

    #optimizer
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=0.1,
        betas=(0.9,0.95)
    )

    scalar = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # checkpoint resume if available
    start_step=0
    if os.path.exists(checkpoint_dir):
        checkpoints=[f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        # checkpoint with highest step
        if len(checkpoints) > 0:
            largest_ckpt=max(checkpoints,key=lambda x:int(x.split("_")[-1].split(".")[0]))
            ckpt_path=os.path.join(checkpoint_dir,largest_ckpt)
            print(f"Resuming from checkpoint: {ckpt_path}")

            checkpoint=torch.load(ckpt_path,map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step=checkpoint["step"]
            print(f"Resumed from step {start_step}. Loss at checkpoint: {checkpoint['loss']:.4f}")





    train_loader=get_dataloader(bin_path="train.bin", 
                                batch_size=batch_size, 
                                context_length=context_length,
                                num_workers=4) # get dataloader

    val_loader=get_dataloader(bin_path="val.bin",
                              batch_size=batch_size,
                              context_length=context_length,
                              num_workers=2) # get val dataloader
    
    model.train() # set model to training mode

    step=start_step
    start_time=time.time()
    

    # the training loop
    for x,y in train_loader:
        if step>max_steps:
            break

        t0=time.time() # track time for each step

        # validation check every 1000 steps
        if step>0 and step%1000==0:
         val_loss=evaluate(model,val_loader,device,floattype)
         print(f"Step: {step:05d}, Validation Loss: {val_loss:.4f}")

            

         # checkpoint saving every 1000 steps
         checkpoint_path=os.path.join(checkpoint_dir,f"baat_model_step_{step:05d}.pth")
         raw_model=model._orig_mod if hasattr(model,"_orig_mod")else model # for compiled model
         torch.save({
             "step":step,
             "model_state_dict":raw_model.state_dict(),
             "optimizer_state_dict":optimizer.state_dict(),
             "loss":loss.item()
         },checkpoint_path)
         print(f"Checkpoint saved at step {step} to {checkpoint_path}")
        
        #setting the lr for step
        lr=get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr

        #move data to GPU, async transfer
        x=x.to(device,non_blocking=True)
        y=y.to(device,non_blocking=True)

        #clear old gradients
        optimizer.zero_grad(set_to_none=True)

        #forward pass withbfloat 16 precision 
        with torch.autocast(device_type=device.type, dtype=floattype):
            logits=model(x) # forward pass
        # flat the tensor to match cross entropy loss form
            loss=f.cross_entropy(logits.view(-1,config.vocab_size),y.view(-1)) 

        #backward pass with gradient scaling
        scalar.scale(loss).backward() 

        #gradient clipping to prevent exploding gradients (not needed for bfloat 16 but still keeping if training on my machine instead)
        scalar.unscale_(optimizer) # unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Grad_cl)


        #update wieghts
        scalar.step(optimizer) # update weights
        scalar.update() # ui[date] scalara

        # metrics
        torch.cuda.synchronize() # wait for GPU to finish
        t1=time.time() # end time for step
        dt=(t1-t0)*1000 # time taken for step in MS
        tokens_per_sec=batch_size*context_length/(t1-t0) # tokens processed per second

        if step%10==0:
            print(f"Step: {step:05d}, Loss: {loss.item():.4f}, LR: {lr:.6f}, Time/Step: {dt:.2f}ms, Tokens/sec: {tokens_per_sec:.2f}")



        step+=1 # increment step

    #saving final model after training
    final_path="baat_model_final.pt"
    raw_model=model._orig_mod if hasattr(model,"_orig_mod")else model # for compiled model
    torch.save(raw_model.state_dict(),final_path)
    print(f"Training completed. Final model saved to {final_path}")
    print(f"Total training time: {(time.time()-start_time)/3600:.2f} hours")


if __name__=="__main__":
    train()
