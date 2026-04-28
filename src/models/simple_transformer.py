import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseModel


class SimpleTransformer(BaseModel):
    """
    A simple Transformer encoder model that mirrors the SimpleNeuralNetwork interface.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_patches: int = 4,
        ff_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if input_size % num_patches != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"num_patches ({num_patches})."
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.patch_size = input_size // num_patches
        ff_dim = ff_dim or hidden_size * 4

        self.input_proj = nn.Linear(self.patch_size, hidden_size)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, feature)
            norm_first=True,    # Pre-LN for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:
        batch = x.size(0)
        x = x.view(batch, self.num_patches, self.patch_size)
        x = self.input_proj(x) + self.pos_embedding   # (batch, num_patches, hidden_size)
        x = self.encoder(x)                           # (batch, num_patches, hidden_size)
        x = x.mean(dim=1)                             # (batch, hidden_size)
        x = self.fc_out(x)                            # (batch, output_size)
        return self.sigmoid(x)