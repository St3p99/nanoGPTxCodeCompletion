import os
from transformers import GPT2Tokenizer
import numpy as np
import json
import pickle

lits = json.load(open("literals.json"))
tokenizer_dir=""
tokenizer_class = GPT2Tokenizer
out_dir="../../out-token-completion"
def get_special_tokens(path):
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens


def prepare(file_type):  
    with open(os.path.join(os.path.dirname(__file__), f"{file_type}.txt"), 'r') as f:
        data = f.read()

    # encode with tiktoken gpt2 bpe
    # enc = tiktoken.get_encoding("gpt2")
    # ids = enc.encode_ordinary(data)
    
    # get special tokens
    special_tokens = get_special_tokens(lits)
    tokenizer = tokenizer_class.from_pretrained(
       "gpt2", 
       sep_token='<EOL>', 
       bos_token='<s>', 
       eos_token='</s>', 
       pad_token='<pad>', 
       unk_token='<|UNKNOWN|>', 
       additional_special_tokens=special_tokens
    )
    with open('../../out-token-completion/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ids = tokenizer.encode(data)
    print(f"{file_type} has {len(ids):,} tokens")

    # export to bin files
    ids = np.array(ids, dtype=np.uint16)
    
    ids.tofile(os.path.join(os.path.dirname(__file__), f'{file_type}.bin'))
    


def main():

    prepare(file_type="train")
    prepare(file_type="val")

if __name__ == "__main__":
    main()