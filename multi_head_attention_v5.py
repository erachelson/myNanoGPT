import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # 384/6 = 64, so head_size = 64
n_layer = 6 # number of blocks stacked sequentially
n_head = 6 # number of heads per block
dropout = 0.2

# seeding (for reproducibility)
torch.manual_seed(1337)

# open text file
filename = 'tinyshakespeare.txt' # 'tinymoliere.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

# tokens
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode and decode strings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encode a string as a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers as a string

# split dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

# estimate training and validation losses (for monitoring purposes)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# single attention head
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)   # B,T,C warning, here C stands for head_size
        q = self.query(x) # B,T,C
        # compute affinities and attention scores
        wei = q @ k.transpose(-2,-1) * C**-.5 # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # aggregate values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ ReLU layer """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication then computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # skip connection
        x = x + self.ffwd(self.ln2(x)) # skip connection
        return x

# Single attention head language model
class MultiHeadAttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The logits for the next token are simply stored in a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T  = idx.shape
        # idx and targets are both (B,T)-shaped tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) 
        # warning here, T might grow larger than block_size if we generate more tokens than block_size, which will cause an index error in the previous line, but we can ignore that for now.
        x = tok_emb + pos_emb # the sum broadcasts the (T,C) tensor across batches
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # at this point, logits are a (B,T,C)-shaped tensor
        # Pytorch expects a (n_data,n_classes) tensor to compute a CE loss
        # so we reshape the logits if we need to compute a loss
        # otherwise, keep the logits in the shape they are
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) array of indices
        for _ in range(max_new_tokens):
            # get prediction
            idx_cond = idx[:,-block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to sequence
            idx = torch.cat((idx,idx_next), dim=1)
        return idx

model = MultiHeadAttentionLanguageModel()
m = model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    # periodically evaluate the validation loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample minibatch
    xb, yb = get_batch('train')
    # evaluate loss and take gradient step
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from model
idx = torch.zeros((1,1), dtype=torch.long, device=device)
new_seq = m.generate(idx, max_new_tokens=500)
print("predicted sequence:\n", new_seq)
print("... as a string of characters:\n", decode(new_seq[0].tolist()))
