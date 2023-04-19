"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import pickle

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-token-completion' # ignored if init_from is not 'resume'

#5535
start = '<s> package net . codejava . io ; import java . io . FileInputStream ; import java . io . IOException ; import java . io . InputStreamReader ; public class TextFileReadingExample2 { public static void main ( String [ ] args) { try { FileInputStream inputStream = new FileInputStream ( " MyFile . txt " ) ; InputStreamReader reader ='
# or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"

num_samples = 5 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 10 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load tokenizer
tokenizer = None
load_tokenizer = False
if init_from == 'resume':
    tokenizer_path = os.path.join(out_dir, 'tokenizer.pkl')
    load_tokenizer = os.path.exists(tokenizer_path)
if load_tokenizer:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    encode = lambda s: tokenizer.encode(s)
    decode = lambda l: tokenizer.decode(l)
else:
    # ok let's assume gpt-2 encodings by default
    print("No tokenizer.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def DecodeIds(idxs):
    codes = ""
    for idx in idxs:
        to_add = tokenizer.convert_ids_to_tokens(idx)
        if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
            if not codes.endswith(" "):
                codes += " " + to_add[1:]
            else:
                codes += to_add[1:]
        elif (
            idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
            tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
        ):
            codes += " " + to_add + " "
        else:
            codes += to_add
    return codes.strip(" ")


print('----PROMPT-----')
print(start)
print('---------------')
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # output = decode(y[0].tolist())
            # print(output[len(output) - 1000:])
            print(f'---SAMPLE--{k}---')
            print(DecodeIds(y[0].tolist()))
            print('---------------')
