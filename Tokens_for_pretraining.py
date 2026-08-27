import numpy as np
import time
from datasets import load_dataset
from tokenizers import Tokenizer

# reusing the clean "None" val error fixer function
def valid_text(iterator,key):
    while True:
        try:
            doc=next(iterator)
            val=doc.get(key)
            if val is not None and isinstance(val,str) and val.strip() != "":
                return val
        except StopIteration:
            return None

# val and training split
def write_tokens_to_bin(filename,target_tokens,urdu_iter,rom_iter,eng_iter,tokenizer,eot_token):
    tokens_written =0
    start_time=time.time()
    last_tok_written=0

    
    print(f"starting the process for {filename} ")
        
    with open (filename,"wb") as f:
        while tokens_written<target_tokens:
            chunk_tokens=[]

    
            #70% Formal urdu
            for _ in range(70):
                text=valid_text(urdu_iter,"text")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)
    
            #15% Roman Urdu
            for _ in range(15):
                text=valid_text(rom_iter,"message")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)
                        
            #15% English
            for _ in range(15):
                text=valid_text(eng_iter,"text")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)
                
            #writing to disc
            np.array(chunk_tokens, dtype=np.uint16).tofile(f)
            tokens_written+=len(chunk_tokens)
    
            #printing every 1M successfull pass
            if tokens_written-last_tok_written >= 1000000:
                elapsed=time.time()-start_time
                print(f"Progress update: {tokens_written/1000000:.1f}M out of 2B tokens.| Time elapsed {elapsed:.1f}s")
                last_tok_written=tokens_written
    
        print(f"Data Prep completed for {filename}, Final Token count: {tokens_written}")

# preparing the dataset
def prepare_dataset():
    print("Loading //.")
    tokenizer=Tokenizer.from_file("tokenizer.json")
    eot_token=tokenizer.token_to_id("[endoftext]")

    urdu_iter=iter(load_dataset("allenai/c4", "ur", split="train", streaming=True))
    rom_iter=iter(load_dataset("Khubaib01/RomanUrdu-NLP-Sentiment-Corpus", split="train", streaming=True))
    eng_iter=iter(load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True))

    # gonig for 4.2B tokens because of the chinchilla Scaling Laws, as the arch is approx 210 now10 approx so means 210Mx20=2B approx
    # split as 40 mil for val and 4.16B for actual training
    # val 

    val_tokens=100_000 # 40_000_000 #40M tokens for validation
    write_tokens_to_bin("val.bin",val_tokens,urdu_iter,rom_iter,eng_iter,tokenizer,eot_token)
    # training
    train_tokens=500_000 # 4_160_000_000 # 4.16B tokens for training (total 4.2B tokens)
    write_tokens_to_bin("train.bin",train_tokens,urdu_iter,rom_iter,eng_iter,tokenizer,eot_token)
  


if __name__=="__main__":
    prepare_dataset()   