import torch
import torch.nn as nn
from .attention import RoPEMultiheadAttention, MultiheadCrossAttention
from .transformers import TransformerBlock, TransformerDecoderBlock, TransformerEncoder, TransformerDecoder, TinyLlamaTransformer
from config import TinyAutoencoderConfig, TinyKoopmanAutoencoderConfig, TinyDVAEConfig

class TransformerAutoencoder(nn.Module):
    # def __init__(self, vocab_size, embed_dim, latent_dim, n_layers, n_heads, ffn_dim, context_window, cls_id, sos_id, n_latents=8, latent_dropout=0.03):
    def __init__(self, config: TinyAutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config.encoder_config)
        self.decoder = TransformerDecoder(config.decoder_config)
        self.latent_dropout = nn.Dropout(config.dropout_latent)

        self.cls_id = config.cls_id
        self.sos_id = config.sos_id
        self.context_window = config.context_window

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
        
class TransformerDVAE(nn.Module):
    def __init__(self, config: TinyDVAEConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        
        # Encoder (inference) model
        self.encoder = TinyLlamaTransformer(config.encoder_config)
        self.encoder_ln = nn.LayerNorm(config.embed_dim)
        
        # Decoder (emission) model
        # self.dropout_decoder = nn.Dropout(config.dropout_decoder)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_decoder),
            nn.Linear(config.decoder_ffn_dim, len(config.vocab))
        )
        self.decoder_ln = nn.LayerNorm(config.latent_dim) if config.decoder_ln else None

        # Transition model
        # self.dropout_transition = nn.Dropout(config.dropout_transition)
        self.transition = nn.Sequential(
            nn.Linear(config.latent_dim, config.transition_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_transition),
            nn.Linear(config.transition_ffn_dim, config.latent_dim * 2)
        )
        self.transition_ln = nn.LayerNorm(config.latent_dim) if config.transition_ln else None

        self.transition_t0 = nn.Sequential(
            nn.Linear(config.latent_dim, config.transition_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_transition),
            nn.Linear(config.transition_ffn_dim, config.latent_dim * 2)
        )
        self.transition_ln_t0 = nn.LayerNorm(config.latent_dim) if config.transition_ln else None

        self.latent_dropout = nn.Dropout(config.dropout_latent)

        self.context_window = config.context_window
        self.pooling = config.pooling

        # additional layers to produce mean and logvar for VAE
        self.to_mu = nn.Linear(config.encoder_config.embed_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.encoder_config.embed_dim, config.latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def inference(self, x, return_internals=False):
        "Run the inference model to get latent representation z, mean and logvar"
        B, T = x.shape
        if return_internals:
            enc_out, initial_embeddings, final_embeddings = self.encoder(x, return_internals=return_internals, use_head=False)
        else:
            enc_out = self.encoder(x, return_internals=return_internals, use_head=False) # (B, T, E)
        
        if self.pooling == 'last':
            latent_repr = enc_out[:, -1, :] # (B, E)
        elif self.pooling == 'mean':
            latent_repr = enc_out.mean(dim=1) # (B, E)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # apply LayerNorm
        latent_repr = self.encoder_ln(latent_repr) # (B, E)

        mu = self.to_mu(latent_repr) # (B, latent_dim)
        logvar = self.to_logvar(latent_repr) # (B, latent_dim)
        z = self.reparameterize(mu, logvar) # (B, latent_dim)

        if return_internals:
            return z, mu, logvar, initial_embeddings, final_embeddings
        else:
            return z, mu, logvar

    def transition_model(self, z, is_t0=False):
        "Predict the distribution of the next latent state given the current latent state"
        # z is of shape (B, latent_dim)
        if is_t0:
            if self.transition_ln_t0:
                z = self.transition_ln_t0(z)
            mu_logvar = self.transition_t0(z) # (B, latent_dim * 2)
        else:
            if self.transition_ln:
                z = self.transition_ln(z)
            mu_logvar = self.transition(z) # (B, latent_dim * 2)
        mu, logvar = mu_logvar.chunk(2, dim=-1) # each of shape (B, latent_dim)
        return mu, logvar

    def decode(self, z):
        "Decode latent representation z to logits over vocabulary"
        # z is of shape (B, latent_dim)
        if self.decoder_ln:
            z = self.decoder_ln(z)
        logits = self.decoder(z) # (B, vocab_size)
        return logits

#TODO: Complete LowRankTransition module
# add the correct mean computation with the decay term and the activation in between
# add the variance prediction
# add time discretization factors?
class LowRankTransition(nn.Module):
    def __init__(self, latent_dim, rank, init_scale=0.01, decay=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.rank = rank
        # TODO: add option for uniform, correlated gaussian initializations
        self.A = nn.Parameter(torch.randn(latent_dim, rank) * init_scale)  # Low-rank factor A
        self.B = nn.Parameter(torch.randn(rank, latent_dim) * init_scale)  # Low-rank factor B
        self.bias = nn.Parameter(torch.zeros(latent_dim))  # Bias term
        self.logvar = nn.Parameter(torch.ones(latent_dim))  # Diagonal covariance for Gaussian noise

        self.decay_param = nn.Parameter(torch.log(-torch.log(torch.ones(1) * decay)))

    @property
    def decay(self):
        return torch.exp(-torch.exp(self.decay_param))

    def forward(self, z):
        """
        z: (B, latent_dim)
        Returns:
            next_mu: (B, latent_dim) - mean of the next latent state
            next_logvar: (B, latent_dim) - log-variance of the next latent state
        """
        # Compute the low-rank transition
        transition_matrix = torch.matmul(self.A, self.B)  # (latent_dim, latent_dim)
        next_mu = torch.matmul(z, transition_matrix) + self.bias  # (B, latent_dim)
        next_logvar = torch.clamp(2 * self.logvar, min=1e-6, max=100)

        return next_mu, next_logvar

def kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """
    Computes the KL divergence between two multivariate Gaussians with diagonal covariances.
    D_KL(q || p)
    """
    # input shapes: (B, latent_dim)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)

    # Term 1: log(det(Sigma_p) / det(Sigma_q))
    # For diagonal matrices, det(Sigma) = product of diagonal elements.
    # log(det(Sigma)) = sum of log of diagonal elements.
    # The diagonal of Sigma is the variance vector.
    # So, log(det(Sigma)) = sum(log(var)) = sum(logvar).
    term1 = torch.sum(logvar_p - logvar_q, dim=1)  # shape (B,)

    # Term 2: tr(Sigma_p^-1 * Sigma_q)
    # For diagonal matrices, this is the sum of (var_q / var_p).
    term2 = torch.sum(var_q / var_p, dim=1)  # shape (B,)
    
    # Term 3: (mu_p - mu_q)^T * Sigma_p^-1 * (mu_p - mu_q)
    # For diagonal matrices, this is the sum of ((mu_p - mu_q)^2 / var_p).
    term3 = torch.sum(((mu_p - mu_q).pow(2)) / var_p, dim=1)  # shape (B,)
    
    # k is the latent dimension
    k = mu_q.size(1)  # latent_dim 

    # The KL divergence is 0.5 * (term1 - k + term2 + term3)
    kld = 0.5 * (term1 - k + term2 + term3)  # shape (B,)

    return kld

# # --- In your training loop ---
# # Get parameters from both models
# mu_q, logvar_q = encoder(x_t, ...)
# mu_p, logvar_p = transition_model(z_t_minus_1, ...)

# # Compute the KL loss
# KLD_loss = kl_divergence_gaussians(mu_q, logvar_q, mu_p, logvar_p)

# # total_loss = reconstruction_loss + KLD_loss
# # ... and so on