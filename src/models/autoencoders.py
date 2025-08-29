import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --------- 6. Transformer Model (Llama-inspired, RoPE) ---------
# RoPE implementation (from Llama paper and HF)
def apply_rope(x, base=10000.0, seq_dim=1):
    # x: [batch, seq, n_heads, head_dim]
    # RoPE mixes head_dim pairs
    batch, seq, n_heads, head_dim = x.size()
    half_dim = head_dim // 2
    pos = torch.arange(seq, dtype=torch.float32, device=x.device)
    idx = torch.arange(half_dim, dtype=torch.float32, device=x.device)
    freq = torch.exp(-math.log(base) * idx / half_dim)
    angles = pos[:, None] * freq[None, :]
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x_rope = torch.cat([x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :],
                        x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :]], dim=-1)
    return x_rope

def generate_sinusoidal_embeddings(n_latents, embed_dim):
    pe = torch.zeros(n_latents, embed_dim)
    position = torch.arange(0, n_latents, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0) # [1, n_latents, embed_dim]

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, causal_mask=True, dropout=0.03):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        B, S, E = x.shape

        # Create causal mask
        if self.causal_mask:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)

        qkv = self.qkv_proj(x)                # [B, S, 3E]
        q, k, v = qkv.chunk(3, dim=-1)        # [B, S, E] each

        # reshape for multihead: [B, S, n_heads, head_dim]
        def split_heads(t):
            return t.view(B, S, self.n_heads, self.head_dim)
        q, k, v = map(split_heads, (q, k, v))

        # Apply RoPE to q and k
        q, k = apply_rope(q), apply_rope(k)

        # [B, n_heads, S, head_dim]
        q, k, v = [x.permute(0,2,1,3) for x in (q,k,v)]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if self.causal_mask:
            attn_weights = attn_weights + causal_mask[None, None, :, :]  ## MASK ADDED BY CLAUDE
        attn_weights = self.dropout(attn_weights.softmax(dim=-1))  ## DROPOUT ADDED BY CLAUDE

        attn_output = torch.matmul(attn_weights, v)   # [B, n_heads, S, head_dim]
        attn_output = attn_output.permute(0,2,1,3).contiguous().view(B, S, E)
        return self.out_proj(attn_output)

class MultiheadCrossAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim, n_latents, n_heads, dropout=0.03):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, embed_dim * n_latents, bias=False)
        self.v_proj = nn.Linear(latent_dim, embed_dim * n_latents, bias=False)
        # self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.n_latents = n_latents

        # Pre-compute and store sinusoidal positional encodings
        pe = generate_sinusoidal_embeddings(self.n_latents, embed_dim)
        self.register_buffer('positional_encodings', pe)

    def forward(self, x, z):
        # x: [batch, seq, embed_dim]
        # z: [batch, 1, latent_dim]
        B, S, E = x.shape

        q = self.q_proj(x)                # [B, S, E]
        k = self.k_proj(z)                # [B, 1, E * n_latents]
        v = self.v_proj(z)                # [B, 1, E * n_latents]

        k = k.view(B, self.n_latents, E)  # [B, n_latents, E]
        v = v.view(B, self.n_latents, E)  # [B, n_latents, E]
        # kv = kv.view(B, self.n_latents, E * 2)
        # k, v = kv.chunk(2, dim=-1)        # [B, n_latents, E] each
        # q, k, v = qkv.chunk(3, dim=-1)        # [B, S, E] each

        # Add positional encodings to the Keys and Values
        # The positional_encodings tensor is [1, n_latents, embed_dim] and will broadcast
        k = k + self.positional_encodings
        v = v + self.positional_encodings

        # reshape for multihead: [B, S, n_heads, head_dim]
        def split_heads(t):
            return t.view(B, t.size(1), self.n_heads, self.head_dim)
        q, k, v = map(split_heads, (q, k, v))

        # [B, n_heads, S, head_dim]
        q, k, v = [x.permute(0,2,1,3) for x in (q,k,v)]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = self.dropout(attn_weights.softmax(dim=-1))  ## DROPOUT ADDED BY CLAUDE

        attn_output = torch.matmul(attn_weights, v)   # [B, n_heads, S, head_dim]
        attn_output = attn_output.permute(0,2,1,3).contiguous().view(B, S, E)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout=0.03, causal_mask=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, n_heads, causal_mask=causal_mask)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, latent_dim, n_latents, dropout=0.03):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, n_heads, causal_mask=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiheadCrossAttention(embed_dim, latent_dim, n_latents, n_heads)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, z):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.cross_attention(self.ln2(x), z))
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, cls_id, embed_dropout=0.03, normalize_before_projection=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ffn_dim, causal_mask=False) for _ in range(n_layers)
        ])
        self.latent_projection = nn.Linear(embed_dim, latent_dim, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.context_window = context_window
        self.cls_id = cls_id
        self.normalize_before_projection = normalize_before_projection


    def forward(self, x, return_internals=True):
        """
        Forward pass supporting token indices, probability distributions over tokens, or embedded vectors.

        Args:
            x (Tensor):
                - [B, T] if token indices (int)
                - [B, T, vocab_size] if probability distributions
                - [B, T, embed_dim] if pre-computed embeddings
            return_internals (bool): whether to return intermediate embeddings

        Returns:
            logits or (logits, initial_embeddings, final_embeddings)
        """
        B = x.size(0)
        T = x.size(1)
        E = self.embed.embedding_dim
        V = self.embed.num_embeddings

        if T > self.context_window:
            raise ValueError(f"Input sequence length {T} exceeds context window {self.context_window}")

        # Case 1: Token indices [B, T]
        if x.ndim == 2:
            # prepend [CLS] token
            x = torch.concat([torch.full((B, 1), self.cls_id, dtype=torch.long, device=x.device), x], dim=1)  # [B, T+1]
            x = self.embed(x)  # [B, T+1, E]

        # Case 2: Probability distributions over vocabulary [B, T, V]
        elif x.ndim == 3 and x.shape[2] == V:
            # Ensure it sums to 1 along the vocab dimension
            if not torch.allclose(x.sum(dim=2), torch.ones(B, T, device=x.device), atol=1e-4):
                raise ValueError("Input probabilities must sum to 1 along vocab dimension.")
            x = torch.matmul(x, self.embed.weight)  # [B, T, E]

        # Case 3: Precomputed embeddings [B, T, E]
        elif x.ndim == 3 and x.shape[2] == E:
            pass  # already in embedded space

        else:
            raise ValueError(f"Unrecognized input shape: {x.shape}")

        x = self.embed_dropout(x)
        initial_embeddings = x.clone()

        for layer in self.layers:
            x = layer(x)

        if self.normalize_before_projection:
            x = self.layer_norm(x)
        final_embeddings = x.clone()
        latent = self.latent_projection(x[:,0])  # project only the [CLS] token

        if return_internals:
            return latent, initial_embeddings, final_embeddings
        else:
            return latent

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, sos_id, n_latents=8, embed_dropout=0.03):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, n_heads, ffn_dim, latent_dim, n_latents) for _ in range(n_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.context_window = context_window
        self.sos_id = sos_id

    def forward(self, x, z, return_internals=True):
        # x is the decoding seed, shape [B, T]
        B, T = x.shape

        # prepend <SOS> token
        x = torch.concat([torch.full((B, 1), self.sos_id, dtype=torch.long, device=x.device), x], dim=1)  # [B, T+1]

        x = self.embed(x)
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x, z)

        x = self.ln_f(x)
        final_embeddings = x.clone()
        logits = self.head(x)
        if return_internals:
            return logits, final_embeddings
        else:
            return logits

class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, cls_id, sos_id, n_latents=8, latent_dropout=0.03):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, cls_id)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, sos_id, n_latents)
        self.latent_dropout = nn.Dropout(latent_dropout)

        self.cls_id = cls_id
        self.sos_id = sos_id
        self.context_window = context_window

        # weight tying of embeddings and head
        self.decoder.embed.weight = self.encoder.embed.weight
        self.decoder.head.weight = self.encoder.embed.weight

    def forward(self, x, decoding_seed, return_internals=True):
        B, T = x.shape
        latent, initial_embeddings, final_embeddings = self.encoder(x, return_internals=return_internals)
        logits, final_embeddings = self.decoder(decoding_seed, self.latent_dropout(latent), return_internals=return_internals)

        if return_internals:
            return logits, latent, initial_embeddings, final_embeddings
        else:
            return logits, latent
