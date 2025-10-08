import torch
import torch.nn as nn
from .attention import RoPEMultiheadAttention, MultiheadCrossAttention, apply_rope
from .transformers import TransformerBlock, TransformerDecoderBlock, TransformerEncoder, TransformerDecoder, TinyLlamaTransformer, ResidualMLP
from config import TinyAutoencoderConfig, TinyKoopmanAutoencoderConfig, TinyDVAEConfig
from torch.distributions.multivariate_normal import MultivariateNormal

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
        self.encoder_ln = nn.LayerNorm(config.embed_dim * config.context_window) if config.pooling == 'none' else nn.LayerNorm(config.embed_dim) 
        # additional layers to produce mean and logvar for VAE
        # self.to_mu = nn.Linear(config.encoder_config.embed_dim, config.latent_dim)
        self.to_mu = nn.Linear(config.encoder_config.embed_dim * config.context_window, config.latent_dim) if config.pooling == 'none' else nn.Linear(config.encoder_config.embed_dim, config.latent_dim)
        # self.to_mu = nn.Sequential(
        #     nn.Linear(config.encoder_config.embed_dim, config.encoder_config.ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.encoder_config.ffn_dim, config.latent_dim)
        # )
        # self.to_logvar = nn.Linear(config.encoder_config.embed_dim, config.latent_dim)
        # self.to_logvar = nn.Linear(config.encoder_config.embed_dim * config.context_window, config.latent_dim) if config.pooling == 'none' else nn.Linear(config.encoder_config.embed_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.encoder_config.embed_dim * config.context_window, config.latent_dim * (config.latent_dim + 1) // 2) if config.pooling == 'none' else nn.Linear(config.encoder_config.embed_dim, config.latent_dim * (config.latent_dim + 1) // 2)
        # self.to_logvar = nn.Sequential(
        #     nn.Linear(config.encoder_config.embed_dim, config.encoder_config.ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.encoder_config.ffn_dim, config.latent_dim)
        # )
        
        # Decoder (emission) model
        # self.dropout_decoder = nn.Dropout(config.dropout_decoder)
        # self.decoder = nn.Sequential(
        #     nn.Linear(config.latent_dim, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout_decoder),
        #     nn.Linear(config.decoder_ffn_dim, len(config.vocab))
        # )
        self.decoder = ResidualMLP(in_dim=config.latent_dim, hidden_dim=config.decoder_ffn_dim, out_dim=len(config.vocab), n_blocks=3)
        self.decoder_ln = nn.LayerNorm(config.latent_dim) if config.decoder_ln else None

        # Transition model
        # self.dropout_transition = nn.Dropout(config.dropout_transition)
        # self.transition = nn.Sequential(
        #     nn.Linear(config.latent_dim, config.transition_ffn_dim),
        #     nn.GELU(),
        #     nn.Dropout(config.dropout_transition),
        #     nn.Linear(config.transition_ffn_dim, config.latent_dim * (1 + 1))
        #     # nn.Linear(config.transition_ffn_dim, config.latent_dim + config.latent_dim * (config.latent_dim + 1) // 2)
        # )
        transition_out_dim = config.latent_dim + config.latent_dim * (config.latent_dim + 1) // 2
        # transition_out_dim = config.latent_dim * (1 + 1)
        self.transition = ResidualMLP(in_dim=config.latent_dim, hidden_dim=config.transition_ffn_dim, out_dim=transition_out_dim, n_blocks=3)
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def inference(self, x, n_layers=None, return_internals=False):
        "Run the inference model to get latent representation z, mean and logvar"
        B, T = x.shape
        if return_internals:
            enc_out, initial_embeddings, final_embeddings = self.encoder(x, return_internals=return_internals, n_layers=n_layers, use_head=False)
        else:
            enc_out = self.encoder(x, return_internals=return_internals, n_layers=n_layers, use_head=False) # (B, T, E)
        
        if self.pooling == 'last':
            latent_repr = enc_out[:, -1, :] # (B, E)
        elif self.pooling == 'mean':
            latent_repr = enc_out.mean(dim=1) # (B, E)
        elif self.pooling == 'rope-mean':
            B, T, E = enc_out.shape
            # n_heads = self.config.encoder_config.n_heads
            n_heads = 1  # use single head for RoPE
            head_dim = E // n_heads
            
            # Reshape for RoPE: [B, T, E] -> [B, T, n_heads, head_dim]
            enc_out_reshaped = enc_out.view(B, T, n_heads, head_dim)
            
            # Apply RoPE
            enc_out_rope = apply_rope(enc_out_reshaped)
            
            # Reshape back: [B, T, n_heads, head_dim] -> [B, T, E]
            enc_out_rope = enc_out_rope.view(B, T, E)
            
            # Average pooling
            latent_repr = enc_out_rope.mean(dim=1) # (B, E)
        elif self.pooling == 'none':
            latent_repr = enc_out.reshape(B, -1) # (B, T * E)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        # apply LayerNorm
        latent_repr = self.encoder_ln(latent_repr) # (B, E)

        mu = self.to_mu(latent_repr) # (B, latent_dim)
        # logvar = self.to_logvar(latent_repr) # (B, latent_dim)
        logvar = self.to_logvar(latent_repr) # (B, latent_dim*(latent_dim+1)/2)
        L = self.build_cholesky_L(logvar)
        dist_q = MultivariateNormal(loc=mu, scale_tril=L)
        z = dist_q.rsample()
        # z = self.reparameterize(mu, logvar) # (B, latent_dim)

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
            # mu_logvar = self.transition(z) # (B, latent_dim * 2)
            mu_logvar = self.transition(z) # (B, latent_dim + latent_dim*(latent_dim+1)/2)
        # mu, logvar = mu_logvar.chunk(2, dim=-1) # each of shape (B, latent_dim)
        mu, logvar = torch.split(mu_logvar, [self.latent_dim, self.latent_dim * (self.latent_dim + 1) // 2], dim=-1) # (B, latent_dim), (B, latent_dim*(latent_dim+1)/2)
        L_t = self.build_cholesky_L(logvar)
        dist_t = MultivariateNormal(loc=mu, scale_tril=L_t)
        z = dist_t.rsample()
        return mu, logvar

    def decode(self, z):
        "Decode latent representation z to logits over vocabulary"
        # z is of shape (B, latent_dim)
        if self.decoder_ln:
            z = self.decoder_ln(z)
        logits = self.decoder(z) # (B, vocab_size)
        return logits

    def generate_latent_trajectory(self, seq_len, device, do_reparameterization = True, z0 = None):
        "Generate a sequence of latent states given an initial latent state z0"
        self.eval()
        with torch.no_grad():
            if z0 is None:
                mu, logvar = self.transition_model(torch.zeros(1, self.latent_dim, device=device), is_t0=True) # (1, latent_dim)
                z0 = self.reparameterize(mu, logvar) # (1, latent_dim)
            else:
                # check if z0 has batch dimension, if not add it
                if z0.dim() == 1:
                    z0 = z0.unsqueeze(0) # (1, latent_dim)
                z0 = z0.to(device)
            B = z0.size(0)
            z_t = z0
            latent_trajectory = [z_t]
            for t in range(1, seq_len):
                mu_t, logvar_t = self.transition_model(z_t)
                L_t = self.build_cholesky_L(logvar_t)
                dist_t = MultivariateNormal(loc=mu_t, scale_tril=L_t)
                z_t = dist_t.rsample()
                # z_t = self.reparameterize(mu_t, logvar_t) if do_reparameterization else mu_t
                latent_trajectory.append(z_t)
            latent_trajectory = torch.stack(latent_trajectory, dim=1) # (B, seq_len, latent_dim)
            decoded_logits = self.decode(latent_trajectory.view(-1, self.latent_dim)) # (B * seq_len, vocab_size)
            decoded_logits = decoded_logits.view(B, seq_len, -1) # (B, seq_len, vocab_size)
        return latent_trajectory, decoded_logits
    
    def build_cholesky_L(self, logvar):
        var_p_log_diag = logvar[:, :self.latent_dim]
        var_p_off_diag = logvar[:, self.latent_dim:]
        positive_diag = torch.exp(var_p_log_diag)
        L = torch.zeros(logvar.shape[0], self.latent_dim, self.latent_dim, device=logvar.device)
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = var_p_off_diag
        L += torch.diag_embed(positive_diag)
        return L

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