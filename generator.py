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


class Generator():
    def __init__(self, init_from='resume', out_dir='out-token-completion',
                       num_samples=5, max_new_tokens=20, temperature=.8,
                       top_k=None, seed=1337, device = 'cuda:0', dtype = 'float16',
                       compile=False):

        # -----------------------------------------------------------------------------
        self._init_from = init_from # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        self.out_dir = out_dir

        self.num_samples = num_samples # number of samples to draw
        self.max_new_tokens = max_new_tokens # number of tokens generated in each sample
        self.temperature = temperature # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = top_k # retain only the top_k most likely tokens, clamp others to have 0 probability
        self.seed = seed
        self.device = device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.dtype = dtype # 'float32' or 'bfloat16' or 'float16'
        self.compile = compile # use PyTorch 2.0 to compile the model to be faster
        self.model = None
        self.tokenizer = None
        self.encode = None
        self.decode = None
        # -----------------------------------------------------------------------------

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        self.load_model()
        self.load_tokenizer()

    def load_tokenizer(self):
        load_tokenizer = False
        if self._init_from == 'resume':
            tokenizer_path = os.path.join(self.out_dir, 'tokenizer.pkl')
            load_tokenizer = os.path.exists(tokenizer_path)
        if load_tokenizer:
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.encode = lambda s: self.tokenizer.encode(s)
            self.decode = lambda l: self.tokenizer.decode(l)
        else:
            # ok let's assume gpt-2 encodings by default
            print("No tokenizer.pkl found, assuming GPT-2 encodings...")
            enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: enc.decode(l)


    def load_model(self):
        if self._init_from == 'resume':
            # init from a model saved in a specific directory
            ckpt_path = os.path.join(self.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
        elif self._init_from.startswith('gpt2'):
            # init from a given GPT-2 model
            self.model = GPT.from_pretrained(self._init_from, dict(dropout=0.0))

        self.model.eval()
        self.model.to(self.device)
        if compile:
            self.model = torch.compile(self.model) # requires PyTorch 2.0 (optional)

    def set_init_from(self, init_from):
        self._init_from = init_from
        self.load_model()
    
    def DecodeIds(self, idxs):
        codes = ""
        for idx in idxs:
            to_add = self.tokenizer.convert_ids_to_tokens(idx)
            if self.tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id] or
                self.tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")


    def generate(self, prompt):
        # encode the beginning of the prompt
        if prompt.startswith('FILE:'):
            with open(prompt[5:], 'r', encoding='utf-8') as f:
                prompt = f.read()
        ids = self.encode(prompt)
        x = (torch.tensor(ids, dtype=torch.long, device=self.device)[None, ...])
        
        # run generation
        samples = []
        with torch.no_grad():
            with self.ctx:
                for k in range(self.num_samples):
                    y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
                    samples.append(self.decode(y[0].tolist()))
        # print(samples)
        return samples