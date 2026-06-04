import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------
# Hyperparameters
# ------------------------------
batch_size = 64
max_seq_len = 64
max_iters = 3000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

dim = 128
n_layers = 4
num_heads = 4
head_k_dim = 16
head_v_dim = 32
dropout = 0.0
norm_eps = 1e-5
conv_size = 4

torch.manual_seed(1337)

# ------------------------------
# Load plain-text corpus
# ------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------------------------
# Character-level vocabulary
# ------------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# ------------------------------
# Train/validation split
# ------------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - max_seq_len, (batch_size,))
    x = torch.stack([src[i : i + max_seq_len] for i in ix])
    y = torch.stack([src[i + 1 : i + max_seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim_local: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim_local))

    def forward(self, x: torch.Tensor):
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_norm.type_as(x) * self.weight


class HeadwiseRMSNorm(nn.Module):
    def __init__(self, head_dim_local: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim_local))

    def forward(self, x: torch.Tensor):
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_norm.type_as(x) * self.weight


class HeadwiseGatedRMSNorm(nn.Module):
    def __init__(self, head_dim_local: int, eps: float):
        super().__init__()
        self.norm = HeadwiseRMSNorm(head_dim_local, eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        return self.norm(x) * F.silu(gate)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            padding=kernel_size - 1,
        )

    def forward(self, x: torch.Tensor):
        y = self.conv(x.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        return F.silu(y)


class GatedDeltaNetMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_dim = num_heads * head_k_dim
        self.value_dim = num_heads * head_v_dim

        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, num_heads, bias=False)
        self.b_proj = nn.Linear(dim, num_heads, bias=False)
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False)

        self.q_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
        self.k_conv = CausalDepthwiseConv1d(self.key_dim, conv_size)
        self.v_conv = CausalDepthwiseConv1d(self.value_dim, conv_size)

        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.o_norm = HeadwiseGatedRMSNorm(head_v_dim, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape

        q = self.q_conv(self.q_proj(x)).view(bsz, seqlen, num_heads, head_k_dim)
        k = self.k_conv(self.k_proj(x)).view(bsz, seqlen, num_heads, head_k_dim)
        v = self.v_conv(self.v_proj(x)).view(bsz, seqlen, num_heads, head_v_dim)
        gate = self.g_proj(x).view(bsz, seqlen, num_heads, head_v_dim)

        q = F.normalize(q.float(), dim=-1, eps=1e-6).type_as(q) / math.sqrt(head_k_dim)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).type_as(k)
        beta = torch.sigmoid(self.b_proj(x))
        g = -torch.exp(self.A_log).view(1, 1, -1) * F.softplus(self.a_proj(x) + self.dt_bias.view(1, 1, -1))

        state = torch.zeros(bsz, num_heads, head_k_dim, head_v_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]

            state = state * torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)
            pred = torch.einsum("bhkv,bhk->bhv", state, k_t)
            v_new = (v_t - pred) * beta[:, t].unsqueeze(-1)
            state = state + torch.einsum("bhk,bhv->bhkv", k_t, v_new)
            out_t = torch.einsum("bhk,bhkv->bhv", q_t, state)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)
        out = self.o_norm(out, gate).reshape(bsz, seqlen, self.value_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GDNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.mixer = GatedDeltaNetMixer()
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.ffn = FeedForward()

    def forward(self, x: torch.Tensor):
        h = x + self.mixer(self.attn_norm(x))
        return h + self.ffn(self.ffn_norm(h))


class GDNLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([GDNBlock() for _ in range(n_layers)])
        self.norm = RMSNorm(dim, norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        h = self.tok_embeddings(idx)
        for layer in self.layers:
            h = layer(h)
        logits = self.output(self.norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GDNLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

log_path = "gdn.c/nano_train_log.txt"
with open(log_path, "w", encoding="utf-8") as logf:
    logf.write("iter\ttrain_loss\tval_loss\n")
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            msg = f"step {iter}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}"
            print(msg)
            logf.write(f"{iter}\t{losses['train'].item():.6f}\t{losses['val'].item():.6f}\n")
            logf.flush()

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_losses = estimate_loss()
    final_msg = f"FINAL: train loss {final_losses['train']:.6f}, val loss {final_losses['val']:.6f}"
    print(final_msg)
    logf.write(f"FINAL\t{final_losses['train'].item():.6f}\t{final_losses['val'].item():.6f}\n")
    logf.flush()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    sample = decode(generated)
    print("SAMPLE:\n" + sample)
    logf.write("SAMPLE:\n" + sample + "\n")
