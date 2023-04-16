import os
import tiktoken
import numpy as np



def prepare(file_type):  
    with open(os.path.join(os.path.dirname(__file__), f"{file_type}.txt"), 'r') as f:
        data = f.read()

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(data)
    print(f"{file_type} has {len(ids):,} tokens")

    # export to bin files
    ids = np.array(ids, dtype=np.uint16)
    
    ids.tofile(os.path.join(os.path.dirname(__file__), f'{file_type}.bin'))
    


def main():
    prepare(file_type="train")
    prepare(file_type="val")

if __name__ == "__main__":
    main()