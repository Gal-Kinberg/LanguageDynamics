import torch
import torch.nn as nn

class SimpleLinearProbe(nn.Module):
    """
    A simple linear probe for binary classification on transformer embeddings.

    This module takes a batch of embeddings (e.g., from a transformer's hidden state)
    and applies a single linear layer to produce a single logit for binary
    classification.
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initializes the linear probe.

        Args:
            embedding_dim (int): The size of the input embedding vectors (E).
        """
        super(SimpleLinearProbe, self).__init__()
        
        # Define the linear layer.
        # Input features: embedding_dim (E)
        # Output features: 1 (a single logit for binary classification)
        self.linear_layer = nn.Linear(embedding_dim, 1)
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the probe.

        Args:
            embeddings (torch.Tensor): A batch of embeddings with shape [B, E],
                                       where B is batch size and E is embedding_dim.

        Returns:
            torch.Tensor: The output logits with shape [B, 1].
                          A sigmoid function can be applied to these logits
                          to get probabilities.
        """
        # Pass the embeddings through the linear layer
        # Input shape: [B, E]
        # Output shape: [B, 1]
        logits = self.linear_layer(embeddings)
        return logits

class SimpleMLPProbe(nn.Module):
    """
    A simple MLP probe for binary classification on transformer embeddings.

    This module takes a batch of embeddings (e.g., from a transformer's hidden state)
    and applies an MLP to produce a single logit for binary
    classification.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """
        Initializes the linear probe.

        Args:
            embedding_dim (int): The size of the input embedding vectors (E).
        """
        super(SimpleMLPProbe, self).__init__()
        
        # Define the linear layer.
        # Input features: embedding_dim (E)
        # Output features: 1 (a single logit for binary classification)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the probe.

        Args:
            embeddings (torch.Tensor): A batch of embeddings with shape [B, E],
                                       where B is batch size and E is embedding_dim.

        Returns:
            torch.Tensor: The output logits with shape [B, 1].
                          A sigmoid function can be applied to these logits
                          to get probabilities.
        """
        # Pass the embeddings through the linear layer
        # Input shape: [B, E]
        # Output shape: [B, 1]
        logits = self.mlp(embeddings)
        return logits