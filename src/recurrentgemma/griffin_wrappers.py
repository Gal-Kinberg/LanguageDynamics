import torch
from torch import nn
from typing import Optional, Tuple, Any
from recurrentgemma.griffin_layers import RGLRU
from recurrentgemma.griffin_modules import RecurrentBlock, ResidualBlock, RecurrentBlockCache, ModifiedRecurrentBlock, \
    gelu


class RGLRUWrapper(nn.Module):
    """Wrapper for RGLRU to provide PyTorch RNN-compatible interface."""

    def __init__(
            self,
            input_size: int,  # d
            hidden_size: int,  # h
            num_layers: int = 1,
            nonlinearity: str = "tanh",  # Not used (RGLRU is linear).
            bias: bool = True,  # Bias is controlled by BlockDiagonalLinear components inside RGLRU.
            batch_first: bool = False,
            dropout: float = 0.0,  # Dropout not handled in RGLRU.
            bidirectional: bool = False,  # RGLRU does not support bidirectional.
    ):
        """
        Initialize the RNNRGLRU wrapper.

        Args:
            input_size: Number of input features.
            hidden_size: Number of features in the hidden state (width in RGLRU).
            num_layers: Number of recurrent layers (RGLRU supports only 1 layer currently).
            bias: If False, the layer does not use bias weights (not applicable here).
            batch_first: If True, input and output tensors are (batch, seq, feature).
            dropout: If non-zero, introduces a Dropout layer on the outputs of each RGLRU layer (not implemented here).
            bidirectional: If True, becomes a bidirectional RNN (not supported by RGLRU).
        """
        super().__init__()

        if num_layers > 1:
            raise ValueError("RGLRU currently supports only one layer.")

        if bidirectional:
            raise ValueError("RGLRU does not support bidirectional operation.")

        self.input_size = input_size  # d
        self.hidden_size = hidden_size  # h
        self.num_layers = num_layers  # 1
        self.batch_first = batch_first  # b

        # Initialize the single RGLRU layer
        self.rglru = RGLRU(
            width=self.input_size,  # d
            num_heads=1,  # Can be adjusted if default is acceptable
            w_init_variance_scale=1.0,
        )

    def forward(
            self, input: torch.Tensor, hx: Optional[torch.Tensor] = None  # bxtxd  # bxh
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RGLRU.

        Args:
            input: Input tensor of shape (seq_len, batch, input_size) if batch_first=False,
                   otherwise (batch, seq_len, input_size).
            hx: Optional initial hidden state of shape (batch, hidden_size). Defaults to zeros.

        Returns:
            Tuple containing:
                - Output tensor of shape (seq_len, batch, hidden_size) if batch_first=False,
                  otherwise (batch, seq_len, hidden_size).
                - Final hidden state tensor of shape (batch, hidden_size).
        """
        if not self.batch_first:
            # Reorder dimensions if batch_first=False
            input = input.transpose(0, 1)

        batch_size, seq_len, _ = input.shape

        # Initialize hidden state hx if not provided
        if hx is None:
            hx = torch.zeros(
                (batch_size, self.hidden_size), device=input.device, dtype=input.dtype
            )

        # Create dummy segment_pos tensor (all zeros, as we don't use segments in standard use).
        segment_pos = torch.zeros(
            (batch_size, seq_len), device=input.device, dtype=torch.long
        )

        # Pass through RGLRU
        output, hn = self.rglru(
            input, segment_pos, hx, return_cache=True
        )  # input_bxtxd, segment_bxt, hx_bxh

        if not self.batch_first:
            # Reorder dimensions back if batch_first=False
            output = output.transpose(0, 1)

        return output, hn


class GriffinRecurrentBlock(nn.Module):
    """
    Wrapper for RecurrentBlock with a PyTorch RNN-compatible interface.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            batch_first: bool = False,
            bidirectional: bool = False,  # Not supported for RecurrentBlock.
            device: str | torch.device | None = None,
    ):
        """
        Initialize the RNNRecurrentBlock wrapper.

        Args:
            input_size: Size of the input features.
            hidden_size: Size of the hidden state (width in RecurrentBlock).
            num_layers: Number of recurrent layers (only 1 is supported for RecurrentBlock).
            batch_first: If True, inputs and outputs are of shape (batch, seq, feature).
        """
        super().__init__()

        if num_layers > 1:
            raise ValueError("RecurrentBlock only supports a single layer.")

        if bidirectional:
            raise ValueError("RecurrentBlock does not support bidirectional operation.")

        # Assign the relevant variables.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.device = device

        self.conv1d_state = None

        # Initialize the RecurrentBlock module.
        self.recurrent_block = ModifiedRecurrentBlock(
            width=self.input_size,
            lru_width=self.hidden_size,
            num_heads=1,  # Defaulting to a single head; can be parameterized.
            device=self.device,
        )

    def forward(
            self, input_bxtxd: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> tuple[tuple[Any, Any], Any]:
        """
        Forward pass through the RecurrentBlock.

        Args:
            input_bxtxd: Tensor of shape (seq, batch, input_size) if batch_first=False,
                   else (batch, seq, input_size). shape: bxtxd
            hx: Optional initial hidden state, defaults to zeros if not provided. shape: 1xbxh

        Returns:
            output: Tensor of shape (seq, batch, hidden_size) if batch_first=False,
                    else (batch, seq, hidden_size). shape: bxtxh
            hn: Last hidden state of shape (batch, hidden_size). shape: 1xbxh
        """
        if not self.batch_first:
            # Adjust input for batch-first mode.
            input_bxtxd = input_bxtxd.transpose(0, 1)

        batch_size, seq_len, _ = input_bxtxd.shape

        # Initialize the hidden state if none was given.
        if hx is None:
            # hx = torch.zeros(
            #     (batch_size, self.hidden_size), device=input_bxtxd.device, dtype=input_bxtxd.dtype
            # )
            input_cache = RecurrentBlock.init_cache(batch_size=batch_size, lru_width=self.hidden_size,
                                                    device=self.device, dtype=input_bxtxd.dtype)
        else:
            # rg_lru_cache = hx[:, :, :self.hidden_size]
            # conv1d_cache = hx[:, :, self.hidden_size:]
            # conv1d_cache = conv1d_cache.reshape(batch_size, self.input_size, self.hidden_size)
            # input_cache = RecurrentBlockCache(rg_lru_state=rg_lru_cache, conv1d_state=conv1d_cache if seq_len == 1 else None)
            input_cache = RecurrentBlockCache(rg_lru_state=hx, conv1d_state=None)

        # Create a segment_pos tensor where the second dimension counts up from 0 to (seq_len - 1).
        if seq_len == 1:
            segment_pos = torch.ones((batch_size, 1), device=input_bxtxd.device)
        else:
            segment_pos = torch.arange(seq_len, device=input_bxtxd.device).unsqueeze(0).expand(batch_size, -1)

        # Forward through the RecurrentBlock.
        output, rg_lru_state_traj_bxtxh, output_cache = self.recurrent_block(input_bxtxd, segment_pos, input_cache,
                                                                             return_cache=True)
        # conv1d_cache_out = output_cache.conv1d_state.reshape(1, batch_size, -1)
        # hn = torch.cat([output_cache.rg_lru_state, conv1d_cache_out], dim=2)
        hn = output_cache.rg_lru_state
        # self.conv1d_state = output_cache.conv1d_state

        if not self.batch_first:
            # Adjust output back to batch-first mode.
            output = output.transpose(0, 1)

        return (output, rg_lru_state_traj_bxtxh), hn

    def compute_fixed_point(self, input_bxtxd: torch.Tensor
                            ) -> tuple[tuple[Any, Any], Any]:
        """
        Forward pass through the RecurrentBlock.

        Args:
            input_bxtxd: Tensor of shape (seq, batch, input_size) if batch_first=False,
                   else (batch, seq, input_size). shape: bxtxd
            hx: Optional initial hidden state, defaults to zeros if not provided. shape: 1xbxh

        Returns:
            output: Tensor of shape (seq, batch, hidden_size) if batch_first=False,
                    else (batch, seq, hidden_size). shape: bxtxh
            hn: Last hidden state of shape (batch, hidden_size). shape: 1xbxh
        """
        if not self.batch_first:
            # Adjust input for batch-first mode.
            input_bxtxd = input_bxtxd.transpose(0, 1)

        batch_size, seq_len, _ = input_bxtxd.shape

        x1_bxtxh = self.recurrent_block.linear_x(input_bxtxd)
        conv1D_cache_input_bxtemporalxh = x1_bxtxh[:, 0, :].unsqueeze(1).repeat(1,
                                                                                self.recurrent_block.conv1d_temporal_width - 1,
                                                                                1)

        # Initialize the hidden state. the RG-LRU cache is not important since we compute the fixed point analytically.
        input_cache = RecurrentBlockCache(rg_lru_state=self.recurrent_block.rg_lru.init_cache(batch_size, self.hidden_size), conv1d_state=conv1D_cache_input_bxtemporalxh)

        # Create a segment_pos tensor which is only ones (no reset)
        segment_pos = torch.ones((batch_size, 1), device=input_bxtxd.device)

        x2_bxtxh, conv1d_state_bxdxh = self.recurrent_block.conv_1d(
            x=x1_bxtxh[:,0:1,:],
            segment_pos=segment_pos,
            cache=None if input_cache is None else input_cache.conv1d_state,
            return_cache=True,
        )

        x3_bxtxh, rg_lru_state_1xbxh, rg_lru_internals = self.recurrent_block.rg_lru(
            x=x2_bxtxh,
            segment_pos=segment_pos,
            cache=None if input_cache is None else input_cache.rg_lru_state,
            return_cache=True,
            return_internals=True,
        )

        rg_lru_fixed_point = rg_lru_internals['normalized_x'] / (1 - rg_lru_internals['a'])

        # y branch.
        y_bxtxh = self.recurrent_block.linear_y(input_bxtxd)
        y_bxtxh = gelu(y_bxtxh)

        output_fixed_point = rg_lru_fixed_point * y_bxtxh[:, 0:1, :]

        return rg_lru_fixed_point, output_fixed_point


class GriffinResidualBlock(nn.Module):
    """
    Wrapper for ResidualBlock with a PyTorch RNN-compatible interface.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            nonlinearity: str = "tanh",  # Not used; depends on ResidualBlock configuration.
            bias: bool = True,  # Bias controlled internally in ResidualBlock.
            batch_first: bool = False,
            dropout: float = 0.0,  # Dropout ignored for ResidualBlock.
            bidirectional: bool = False,  # Not supported in ResidualBlock.
            temporal_type: str = "RECURRENT",  # Used to select temporal block.
            num_heads: int = 1,
            attention_window_size: int = 1,
    ):
        """
        Initialize the RNNResidualBlock wrapper.

        Args:
            input_size: Size of the input features.
            hidden_size: Size of the hidden state (width in ResidualBlock).
            num_layers: Number of recurrent layers (only 1 supported for now).
            batch_first: If True, inputs and outputs are of shape (batch, seq, feature).
            temporal_type: Select between "RECURRENT" or "ATTENTION" for temporal block type.
            num_heads: Number of heads for the attention or RG-LRU operation.
            attention_window_size: The size of the window for the local attention block.
        """
        super().__init__()

        if num_layers > 1:
            raise ValueError("ResidualBlock only supports a single layer.")

        if bidirectional:
            raise ValueError("ResidualBlock does not support bidirectional operation.")

        # Assign relevant variables.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Initialize the ResidualBlock.
        self.residual_block = ResidualBlock(
            width=hidden_size,
            mlp_expanded_width=hidden_size * 4,  # Following a common MLP ratio.
            num_heads=num_heads,
            attention_window_size=attention_window_size,
            temporal_block_type=temporal_type.upper(),  # Either "RECURRENT" or "ATTENTION".
        )

    def forward(
            self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ResidualBlock.

        Args:
            input: Tensor of shape (seq, batch, input_size) if batch_first=False,
                   else (batch, seq, input_size).
            hx: Optional initial hidden state (not utilized by ResidualBlock).

        Returns:
            output: Tensor of shape (seq, batch, hidden_size) if batch_first=False,
                    else (batch, seq, hidden_size).
            hn: Final hidden state (not applicable for ResidualBlock; same as output's last step).
        """
        if self.batch_first:
            # Adjust input for batch-first mode.
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.shape

        # Create default segment positions (all zeros).
        segment_pos = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=input.device
        )

        # Forward through the ResidualBlock.
        output, _ = self.residual_block(
            input, segment_pos, cache=None, return_cache=False
        )

        if self.batch_first:
            # Adjust output back to batch-first mode.
            output = output.transpose(0, 1)

        # Return output as hn for compatibility.
        return output, output[-1]
