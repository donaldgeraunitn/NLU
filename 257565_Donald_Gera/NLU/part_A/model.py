import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        self.scale = self.h_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, L, d_model = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = (q @ k.transpose(-2, -1)) * self.scale
        similarity = similarity.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(similarity, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)

        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.0,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ff_dim) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)

        # With dropout=0.0 this is the baseline. For Part 2.A point 2,
        # set dropout > 0 to activate dropout before both final output layers.
        self.output_dropout = nn.Dropout(dropout)

        self.slot_out = nn.Linear(d_model, slots_size)
        self.intent_out = nn.Linear(d_model, n_intents)

        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx, seq_lens):
        B, L = idx.shape
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        x = self.output_dropout(x)

        slots = self.slot_out(x)

        # CLS is appended as the last real token of every sentence.
        cls_positions = seq_lens - 1
        batch_positions = torch.arange(B, device=idx.device)
        cls_tokens = x[batch_positions, cls_positions]
        intent = self.intent_out(cls_tokens)

        return slots, intent
