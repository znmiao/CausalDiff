import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return x
        return x[:, :, :-self.size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = None if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        return self.norm(self.net(x) + res)


class TemporalPredictor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = n_features
        for i in range(n_layers):
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, 2 ** i, dropout))
            in_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, n_features))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        h = history.transpose(1, 2)
        h = self.tcn(h).transpose(1, 2)
        return self.head(h[:, -1])
