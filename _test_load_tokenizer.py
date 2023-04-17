import pickle

with open('out-token-completion/meta.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print(len(tokenizer))