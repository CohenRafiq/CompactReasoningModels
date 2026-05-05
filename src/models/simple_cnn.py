import torch
from torch import Tensor
from torch import nn
from typing import List, Optional, Tuple, Union


class ConvNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_size: int,
        hidden_channels: List[int],
        kernel_size: Union[int, Tuple[int, ...]],
        dims: int = 2,
        fc_hidden_size: int = 256,
        fc_num_layers: int = 2,
        input_shape: Optional[Tuple[int, ...]] = None,
        pool_every: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dims in (1, 2), "dims must be 1 or 2"

        # Pick dimension-appropriate building blocks
        Conv      = nn.Conv1d       if dims == 1 else nn.Conv2d
        BN        = nn.BatchNorm1d  if dims == 1 else nn.BatchNorm2d
        MaxPool   = nn.MaxPool1d    if dims == 1 else nn.MaxPool2d
        AvgPool   = nn.AdaptiveAvgPool1d if dims == 1 else nn.AdaptiveAvgPool2d

        # ── Convolutional backbone ──────────────────────────────────────────
        padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        conv_blocks: List[nn.Module] = []
        in_ch = input_channels

        for i, out_ch in enumerate(hidden_channels):
            conv_blocks += [
                Conv(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                BN(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool_every and (i + 1) % pool_every == 0:
                conv_blocks.append(MaxPool(kernel_size=2))
            in_ch = out_ch

        # Global average pool → fixed-size representation regardless of input size
        conv_blocks.append(AvgPool(1) if dims == 1 else AvgPool((1, 1)))
        self.conv = nn.Sequential(*conv_blocks)

        # ── Fully-connected head ────────────────────────────────────────────
        fc_in = in_ch  # after global avg pool each spatial dim = 1
        fc_layers: List[nn.Module] = [nn.Flatten(), nn.Linear(fc_in, fc_hidden_size), nn.ReLU(inplace=True)]
        if dropout:
            fc_layers.append(nn.Dropout(dropout))

        for _ in range(fc_num_layers - 1):
            fc_layers += [nn.Linear(fc_hidden_size, fc_hidden_size), nn.BatchNorm1d(fc_hidden_size), nn.ReLU(inplace=True)]
            if dropout:
                fc_layers.append(nn.Dropout(dropout))

        fc_layers += [nn.Linear(fc_hidden_size, output_size), nn.Sigmoid()]
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L) for 1-D  |  (B, C, H, W) for 2-D
        out = self.conv(x)
        out = self.fc(out)
        return out