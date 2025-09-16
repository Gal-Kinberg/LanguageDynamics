import torch
import torch.nn as nn
from .attention import RoPEMultiheadAttention, MultiheadCrossAttention
from config import TinyEncoderConfig, TinyDecoderConfig, TinyLMConfig


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, eos_id, dropout_residual=0.03, dropout_attention=0.03, causal_mask=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, n_heads, eos_id=eos_id, causal_mask=causal_mask, dropout=dropout_attention)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_residual)
        self.eos_id = eos_id

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, latent_dim, n_latents, eos_id, dropout_residual=0.03, dropout_attention=0.03):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, n_heads, eos_id=eos_id, causal_mask=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiheadCrossAttention(embed_dim, latent_dim, n_latents, n_heads, dropout=dropout_attention)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_residual)
        self.eos_id = eos_id

    def forward(self, x, z):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.cross_attention(self.ln2(x), z))
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x

class TransformerEncoder(nn.Module):
    # def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, cls_id, embed_dropout=0.03, normalize_before_projection=True):
    def __init__(self, config: TinyEncoderConfig):
        super().__init__()
        self.config = config
        eos_id = config.vocab.index('<EOS>') if '<EOS>' in config.vocab else -1
        self.embed = nn.Embedding(len(config.vocab), config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout_embed)
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ffn_dim, eos_id=eos_id, causal_mask=False, dropout_residual=config.dropout_residual, dropout_attention=config.dropout_self_attention) for _ in range(config.n_layers)
        ])
        self.latent_projection = nn.Linear(config.embed_dim, config.latent_dim, bias=False)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.context_window = config.context_window
        self.cls_id = config.cls_id
        self.normalize_before_projection = True


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
    # def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, sos_id, n_latents=8, embed_dropout=0.03):
    def __init__(self, config: TinyDecoderConfig):
        super().__init__()
        self.config = config
        eos_id = config.vocab.index('<EOS>') if '<EOS>' in config.vocab else -1
        self.embed = nn.Embedding(len(config.vocab), config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout_embed)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(config.embed_dim, config.n_heads, config.ffn_dim, config.latent_dim, config.n_latent, eos_id=eos_id, dropout_residual=config.dropout_residual, dropout_attention=config.dropout_cross_attention) for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.embed_dim, len(config.vocab), bias=False)
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.context_window = config.context_window
        self.sos_id = config.sos_id

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

class TinyLlamaTransformer(nn.Module):
    # def __init__(self, vocab_size, embed_dim, n_layers, n_heads, ffn_dim, context_window):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        eos_id = config.vocab.index('<EOS>') if '<EOS>' in config.vocab else -1
        self.embed = nn.Embedding(len(config.vocab), config.embed_dim)
        self.pos_embed = None  # RoPE only
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ffn_dim, eos_id=eos_id, dropout_residual=config.dropout_residual, dropout_attention=config.dropout_self_attention) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, len(config.vocab), bias=False)
        self.head.weight = self.embed.weight  # Optional head-embed weight tying
        self.context_window = config.context_window

    def forward(self, x, return_internals=False, use_head=True):
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
            x = self.embed(x)  # [B, T, E]

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

        initial_embeddings = x.clone().detach()

        for layer in self.layers:
            x = layer(x)

        if use_head:
            x = self.ln_f(x)  #TODO: Replace final embeddings to be before the LayerNorm!
            final_embeddings = x.clone().detach()
            logits = self.head(x)

            if return_internals:
                return logits, initial_embeddings, final_embeddings
            else:
                return logits
        
        else:
            final_embeddings = x.clone().detach()
            if return_internals:
                return final_embeddings, initial_embeddings, final_embeddings
            else:
                return final_embeddings

class TinyLlamaRawTransformer(nn.Module):
    # def __init__(self, embed_dim, n_layers, n_heads, ffn_dim, context_window):
    def __init__(self, config: TinyLMConfig):
        super().__init__()
        self.config = config
        eos_id = config.vocab.index('<EOS>') if '<EOS>' in config.vocab else -1
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, config.ffn_dim, eos_id=eos_id, dropout_residual=config.dropout_residual, dropout_attention=config.dropout_self_attention) for _ in range(config.n_layers)
        ])
        self.embed_dim = config.embed_dim
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.context_window = config.context_window

    def forward(self, x, return_internals=False):
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
        E = self.embed_dim

        if T > self.context_window:
            raise ValueError(f"Input sequence length {T} exceeds context window {self.context_window}")

        # Case 3: Precomputed embeddings [B, T, E]
        if x.ndim == 3 and x.shape[2] == E:
            pass  # already in embedded space

        else:
            raise ValueError(f"Unrecognized input shape: {x.shape}")

        initial_embeddings = x.clone()

        for layer in self.layers:
            x = layer(x)

        # final LayerNorm?
        # x = self.ln_f(x)
        # final_embeddings = x.clone()

        if return_internals:
            return x, initial_embeddings
        else:
            return x