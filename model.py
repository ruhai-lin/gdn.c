import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet
from fla.modules import GatedMLP, RMSNorm


@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    num_heads: int = 4
    head_k_dim: int = 72
    head_v_dim: int = 72
    hidden_dim: int = 768
    hidden_ratio: int = 4
    expand_v: float = 1.0
    attn_mode: str = "chunk"
    conv_size: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 256
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.head_k_dim

    def __post_init__(self):
        if self.head_v_dim != self.head_k_dim:
            self.expand_v = self.head_v_dim / self.head_k_dim
        else:
            self.expand_v = 1.0


class GDNBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mixer = GatedDeltaNet(
            hidden_size=args.dim,
            expand_v=args.expand_v,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            mode=args.attn_mode,
            use_gate=True,
            use_short_conv=True,
            conv_size=args.conv_size,
            norm_eps=args.norm_eps,
            layer_idx=layer_idx,
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn = GatedMLP(
            hidden_size=args.dim,
            hidden_ratio=args.hidden_ratio,
            intermediate_size=args.hidden_dim,
            hidden_act="swish",
            fuse_swiglu=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _, _ = self.mixer(self.attn_norm(x))
        h = x + attn_out
        return h + self.ffn(self.ffn_norm(h))


class GDNLM(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([GDNBlock(params, i) for i in range(params.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("mixer.o_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.dropout(self.tok_embeddings(tokens))
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay_params = []
        nodecay_params = []
        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2 or pn.endswith("A_log") or pn.endswith("dt_bias"):
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = sum(p.numel() for p in self.parameters())
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * self.params.max_seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
