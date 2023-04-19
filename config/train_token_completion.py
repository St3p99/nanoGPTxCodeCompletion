
# train a miniature code completion model
out_dir = 'out-token-completion'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'token-completion'
wandb_run_name = 'mini-gpt'

dataset = 'token_completion'
batch_size = 8
block_size = 512 # context of up to 512 previous characters

# toddler GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
# system
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be fastersussuss


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
