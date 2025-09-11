import torch
import torch.nn as nn
import numpy as np
import math

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
    def __init__(self, embed_dim, n_heads, eos_id, causal_mask=True, dropout=0.03):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.eos_id = eos_id

    def forward(self, x):
        # x: [batch, seq, embed_dim]
        B, S, E = x.shape

        # Create causal mask
        if self.causal_mask:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)

        #TODO: create "document mask" by <EOS> tokens
        # find EOS tokens
        # eos_indices = (x == self.eos_id).nonzero()  # shape: [-1, 3]

        # create "document" ranges
        # first range always starts at zero
        # last range always ends at last index
        # if no EOS was found, simply full range is allowed

        # create the mask given the ranges

        # def create_document_attention_mask(doc_boundaries, seq_len):
        #     # Initialize a mask with zeros
        #     mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
            
        #     # Fill the mask based on document boundaries
        #     for i in range(len(doc_boundaries)):
        #         # Get the indices of the current document
        #         start_i, end_i = doc_boundaries[i]
                
        #         # Mark attention within the current document as True
        #         mask[start_i:end_i+1, start_i:end_i+1] = True
            
        #     return mask

        # # Create the mask for our example
        # seq_len = len(concatenated_tokens)
        # doc_mask = create_document_attention_mask(doc_boundaries, seq_len)

        # Now, convert boolean mask to float mask for addition
        # attention_mask_float = doc_mask.float().masked_fill(doc_mask == 0, float('-inf'))

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
