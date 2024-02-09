
import time
import random
import numpy as np

import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

DEBUG = False

def count_parameters(model):
    total_params = 0
    total_embedding_params = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        if "embedding" in name.lower():  # Check if the parameter belongs to an embedding layer
            total_embedding_params += param_count
        total_params += param_count

    total_non_embedding_params = total_params - total_embedding_params
    print(f"Total parameters: {total_params}")
    print(f"Total embedding parameters: {total_embedding_params}")
    print(f"Total non-embedding parameters: {total_non_embedding_params}")

def debug_print(txt):
    if DEBUG:
        print(txt)

SEED        = 1337
BATCH_SIZE  = 10
BLOCK_SIZE  = 256
TRAIN_STEPS = 1000
VAL_STEPS   = int(TRAIN_STEPS * 0.1)
LR          = 1e-3

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

enc = tiktoken.get_encoding("cl100k_base")

device = torch.device("cuda")

with open("enwik8", encoding="utf-8") as f:
    txt = f.read()
# chars = sorted(list(set(txt)))
# vocab_len = len(chars)
vocab_len = enc.n_vocab

# stoi = { ch:i for i, ch in enumerate(chars) }
# itos = { i:ch for i, ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

data = torch.tensor(encode(txt), device=device)
print("data.dtype:", data.dtype)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

xb, yb = get_batch("train")

for b in range(BATCH_SIZE):
    for t in range(BLOCK_SIZE):
        context = xb[b, :t+1]
        target  = yb[b, t]
        # print(f"Context: {context.tolist()}, Target: {target}")

def gen(model, max=100, samples=1):
    for s in range(samples):
        print("Sample:", s, "\n",
            decode(
                model.to("cuda").generate(
                    torch.zeros(
                        (1, 1), dtype=torch.long, device="cuda"
                    ),
                    max_new_tokens=max
                )[0].tolist()
            )
        )

class Head(nn.Module):
    def __init__(self, head_size, n_embed, BLOCK_SIZE, dropout):
        super().__init__()
        self.k = nn.Linear(n_embed, head_size, bias=False)
        self.q = nn.Linear(n_embed, head_size, bias=False)
        self.v = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril",
                             torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.k(x)
        q = self.q(x)
        w = q @ k.transpose(-2, -1) * C ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.v(x)
        o = w @ v
        return o

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, BLOCK_SIZE, dropout)
             for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        # print("MHSA proj.shape:", n_embed)
    def forward(self, x):
        # print("MHSA x.shape:", x.shape)
        o = torch.cat([h(x) for h in self.heads], dim=-1)
        # print("MHSA concat o.shape:", o.shape)
        o = self.dropout(self.proj(o))
        # print("MHSA project o.shape:", o.shape)
        return o
    
class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa   = MultiHeadAttention(n_embed, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1  = nn.LayerNorm(n_embed)
        self.ln2  = nn.LayerNorm(n_embed)
    def forward(self, x):
        debug_print("FIRST BLOCK?")
        x = x + self.sa(self.ln1(x))
        debug_print("FIRST SA?")
        x = x + self.ffwd(self.ln2(x))
        debug_print("FIRST FFWD?")
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_len, n_embed, n_heads, n_layer, dropout=0.2):
        super().__init__()
        self.token_emb_table    = nn.Embedding(vocab_len, n_embed)
        self.position_emb_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_heads, dropout=dropout)
              for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_len)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        debug_print("PREPROCESSING")

        tok_embed = self.token_emb_table(idx)
        pos_embed = self.position_emb_table(
            torch.arange(T, device=device))
        
        debug_print("PAST EMBEDS")

        x = tok_embed + pos_embed
        x = self.blocks(x)

        debug_print("PAST ATTN")

        x = self.ln_f(x)
        logits = self.lm_head(x)

        debug_print("WE'RE GETTING PAST PROCESSING")

        if not targets is None:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model     = TransformerDecoder(vocab_len, 384, n_heads=6, n_layer=6).to(device)
count_parameters(model)
out, loss = model(xb, yb)

model = model.to(device) # .to(torch.bfloat16)  # Move model to bfloat16
optim = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = GradScaler()  # Initialize the gradient scaler for AMP

t = time.time()
s = t

for epoch in range(TRAIN_STEPS):
    optim.zero_grad(set_to_none=True)
    
    with autocast(enabled=True, dtype=torch.bfloat16):  # Enable AMP
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device) # Convert data to bfloat16 as appropriate
        logits, loss = model(xb, yb)
    
    scaler.scale(loss).backward()  # Scale the loss to adjust for the reduced precision
    scaler.step(optim)  # Update optimizer
    scaler.update()  # Prepare for the next iteration

    if epoch % 100 == 0:
        cur_epoch_tm = time.time() - t
        t            = time.time()
        total_tm     = time.time() - s
        print(epoch, loss.item(), int(cur_epoch_tm), int(total_tm))
        # gen(model, max=1000)

# model.eval()
# v_losses = []
# for epoch in range(VAL_STEPS):    
#     with autocast(enabled=True, dtype=torch.bfloat16):  # Enable AMP
#         xb, yb = get_batch("val")
#         xb, yb = xb.to(device), yb.to(device) # Convert data to bfloat16 as appropriate
#         logits, loss = model(xb, yb)
#         v_losses.append(loss)

# v_loss = np.array(v_losses).mean()
# print("VAL LOSS:", v_loss)
gen(model, max=1000, samples=3)