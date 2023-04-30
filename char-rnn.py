import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 200
learning_rate = 6e-3
eval_iters = 20
n_embed = 256
n_layer = 2
dropout = 0.2
rnn_type='LSTM'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.manual_seed(42)

data_dir = 'data/tinyshakespeare/'
with open(data_dir + 'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, _, loss = model(X, model.init_hidden(batch_size), Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class charRNN(nn.Module):
    def __init__(self, n_embed, n_hidden, n_layer, dropout=0.5, rnn_type='GRU', tie_weights=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_layer = n_layer

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Embedding(vocab_size, n_embed)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(n_embed, n_hidden, n_layer, dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.RNN(n_embed, n_hidden, n_layer, nonlinearity=rnn_type, batch_first=True)
        self.decoder = nn.Linear(n_hidden, vocab_size)
        
        if tie_weights:
            if n_embed != n_hidden:
                raise ValueError('When using the tied flag, embed size must be equal to hidden size')
            self.decoder.weight = self.encoder.weight
    
    def forward(self, idx, hx, targets=None):
        tok_emb = self.dropout(self.encoder(idx))
        y, h = self.rnn(tok_emb, hx)
        y = self.dropout(y)
        logits = self.decoder(y)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, h, loss
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.n_layer, batch_size, self.n_hidden).to(device)
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.n_layer, batch_size, self.n_hidden).to(device)
            return (h0, c0)
        else:
            return h0
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, hidden, loss = self(idx_cond, self.init_hidden(1))
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = charRNN(n_embed, n_embed, n_layer, rnn_type=rnn_type).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iteration}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # init hidden state for first input
    hb = model.init_hidden(batch_size)
    
    # evaluate the loss
    optimizer.zero_grad()
    logits, hb, loss = model(xb, hb, targets=yb)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
