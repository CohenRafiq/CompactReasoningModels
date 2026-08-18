import torch
from torch import Tensor
from torch import nn
from typing import Optional


class CluePositionalEmbedding(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size))
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed


class GridMLP(nn.Module):
    require_flat_input = True

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.combined_size = hidden_size + output_size

        self.register_buffer("grid", torch.zeros(1, output_size))

        self.clue_pos_embed = CluePositionalEmbedding(input_size)

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                ResidualBlock(
                    output_size=output_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            )

    def forward(self, x: Tensor, layer_num: Optional[int] = None) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2-D input, got {x.dim()}D")
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {x.size(-1)}"
            )

        x = self.clue_pos_embed(x)

        context = self.input_proj(x)                       # (batch, hidden_size)
        batch_grid = self.grid.expand(x.size(0), -1)       # (batch, output_size)
        out = torch.cat([batch_grid, context], dim=-1)    # (batch, combined)

        blocks = self.blocks if layer_num is None else self.blocks[:layer_num]
        for block in blocks:
            out = block(out)

        return out[:, : self.output_size]


class ResidualBlock(nn.Module):

    def __init__(self, output_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.combined_size = output_size + hidden_size

        self.thinking_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(self.combined_size, self.combined_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # Split into grid and thinking components
        grid = x[:, : self.output_size]           # (batch, output_size)
        thinking = x[:, self.output_size :]       # (batch, hidden_size)

        # Normalize only the thinking component
        thinking = self.thinking_norm(thinking)

        # Recombine and pass through the sublayer
        out = torch.cat([grid, thinking], dim=-1)
        out = self.fc(out)
        out = self.activation(out)
        out = self.dropout(out)

        return residual + out