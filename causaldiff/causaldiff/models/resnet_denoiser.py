import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ConvResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.time = nn.Linear(time_dim, channels)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(temb).unsqueeze(-1)
        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return x + h


class PackedResNetDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        n_blocks: int = 8,
        time_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.in_proj = nn.Conv1d(1, hidden_size, 3, padding=1)
        self.blocks = nn.ModuleList([ConvResBlock(hidden_size, time_dim, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(8 if hidden_size % 8 == 0 else 1, hidden_size),
            nn.SiLU(),
            nn.Conv1d(hidden_size, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_proj(z.unsqueeze(1))
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h).squeeze(1)
