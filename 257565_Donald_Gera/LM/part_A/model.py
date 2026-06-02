import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard causal multi-head self-attention used by the baseline model.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        # Each head processes an equal slice of the model representation.
        self.h_dim = d_model // n_heads

        # Scale the dot products to keep attention scores stable before softmax.
        self.scale = self.h_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, L, d_model = x.size()

        # Project the same hidden states into queries, keys, and values.
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split the representation into independent attention heads.
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = (q @ k.transpose(-2, -1)) * self.scale
        # The lower-triangular mask prevents each token from seeing future tokens.
        similarity = similarity.masked_fill(mask == 0, torch.finfo(similarity.dtype).min)

        attn = F.softmax(similarity, dim=-1)

        # Combine values within each head and merge heads back into one representation.
        y = attn @ v
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, L, d_model)
        y = self.out_proj(y)

        return y


# Position-wise feed-forward network applied after attention.
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


# Pre-normalization Transformer block with residual connections.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim)

    def forward(self, x, mask):
        # Residual connections preserve the previous representation around each sub-layer.
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


# Attention variant that applies dropout to attention weights and projected outputs.
class MultiHeadAttentionDropout(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        # Scale the dot products to keep attention scores stable before softmax.
        self.scale = self.h_dim ** -0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, d_model = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = (q @ k.transpose(-2, -1)) * self.scale
        similarity = similarity.masked_fill(mask == 0, torch.finfo(similarity.dtype).min)

        attn = F.softmax(similarity, dim=-1)
        # Regularize which previous positions contribute to the current token.
        attn = self.attn_drop(attn)

        y = attn @ v
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, L, d_model)
        y = self.out_proj(y)
        # Regularize the attention output before the residual addition.
        y = self.proj_drop(y)

        return y


# Feed-forward variant with dropout after the second projection.
class FeedForwardDropout(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Transformer block using the regularized attention and feed-forward variants.
class TransformerBlockDropout(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionDropout(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForwardDropout(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


# Shared GPT-2-style decoder architecture configured by the wrapper classes below.
class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        pos_emb_size,
        d_model,
        n_heads,
        num_layers,
        ff_dim,
        dropout,
        use_dropout,
        weight_tying,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # Learned token and absolute positional embeddings are added before the blocks.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        # The experiment selects either baseline blocks or dropout-enabled blocks.
        if use_dropout:
            self.emb_drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [TransformerBlockDropout(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, ff_dim) for _ in range(num_layers)]
            )

        self.ln_f = nn.LayerNorm(d_model)

        if weight_tying:
            # Weight tying reuses token embeddings as output classifiers, reducing parameters.
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.token_embed.weight
        else:
            self.lm_head = nn.Linear(d_model, vocab_size)

        # Build the causal mask once and slice it to the sequence length during forward passes.
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        B, L = idx.shape
        assert L <= self.pos_emb_size, f"Sequence length {L} is greater than pos_emb_size={self.pos_emb_size}"

        # Every position receives a learned positional embedding.
        pos = torch.arange(L, device=idx.device)

        x = self.token_embed(idx) + self.pos_embed(pos)

        if hasattr(self, "emb_drop"):
            x = self.emb_drop(x)

        # Use only the portion of the precomputed mask needed by this batch.
        mask = self.mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


# Baseline: no dropout and independent input/output embedding matrices.
class GPT2Baseline(GPT2):
    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pos_emb_size=pos_emb_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_dropout=False,
            weight_tying=False,
        )


# Regularized model: enable dropout without weight tying.
class GPT2Dropout(GPT2):
    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.2,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pos_emb_size=pos_emb_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_dropout=True,
            weight_tying=False,
        )


# Parameter-sharing model: enable weight tying without dropout.
class GPT2WeightTying(GPT2):
    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pos_emb_size=pos_emb_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_dropout=False,
            weight_tying=True,
        )


# Combined model: enable both dropout and weight tying.
class GPT2DropoutWeightTying(GPT2):
    def __init__(
        self,
        vocab_size,
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.2,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pos_emb_size=pos_emb_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            use_dropout=True,
            weight_tying=True,
        )
