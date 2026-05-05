import torch
from torch import Tensor
from torch import nn

class SimpleNeuralNetwork(nn.Module):
    """
    A simple feedforward neural network for nonogram solving.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        self.fc2 = nn.Linear(hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            out = layer(out)
            out = batch_norm(out)
            out = self.relu(out)
            out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out