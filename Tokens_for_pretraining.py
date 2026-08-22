import numpy as np
import time
from datasets import load_dataset
from tokenizers import Tokenizer

# reusing the clean None fixer function
def valid_text(iterator,key):
    while True:
        try:
            doc=next(iterator)
            val=doc.get(key)
            if val is not None and isinstance(val,str) and val.strip() != "":
                return val
        except StopIteration:
            return None


# preparing the dataset
def prepare_dataset():
    print("Loading //.")
    tokenizer=Tokenizer.from_file("tokenizer.json")
    eot_tokeninzer=tokenizer.token_to_id("[endoftext]")

    urdu_iter=iter(load_dataset("allenai/c4", "ur", split="train", streaming=True))
    rom_iter=iter(load_dataset("Khubaib01/RomanUrdu-NLP-Sentiment-Corpus", split="train", streaming=True))
    eng_iter=iter(load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True))

    # gonig for 2B tokens because of the chinchilla Scaling Laws, as the arch is 110 approx so means 110Mx20=2B approx
    target_tokens=2_000_000_000
    tokens_written =0
    start_time=time.time()

    print("starting the process for 2B tokens in the bin file train.bin..")
    
    with open ("train.bin","wb") as f:
        while tokens_written<target_tokens:
            chunk_tokens=[]

            #70% Formal urdu
            for _ in range(70):
                text=valid_text(urdu_iter,"text")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)

            for _ in range(15):
                text=valid_text(rom_iter,"message")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)
                    
            for _ in range(15):
                text=valid_text(eng_iter,"text")
                if text:
                    chunk_tokens.extend(tokenizer.encode(text).ids)
                    chunk_tokens.append(eot_token)
            
            #writing to disc
            np.array(chunk_tokens, dtype=np.uint16).tofile(f)
            tokens_written+=len(chunk_tokens)

            #printing every 1M successfull pass
            if tokens_written% 100000<50000:
                elapsed=time.time()-start_time
                print(f"Progress update: {tokens_written/100000:.1f}M out of 2B tokens.| Time elapsed {elapsed:.1f}s")
                print(f"Data Prep completed! Final Token count: {tokens_written}")                


if __name__=="__main__":
    prepare_dataset()