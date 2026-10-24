import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

# bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The logits for the next token are simply stored in a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T)-shaped tensors of integers
        logits = self.token_embedding_table(idx)
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
            # get prediction for last column of idx
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # append to sequence
            idx = torch.cat((idx,idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
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
