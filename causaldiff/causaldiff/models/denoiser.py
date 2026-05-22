import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, time_width: int, dropout: float) -> None:
        super().__init__()
        self.time = nn.Linear(time_width, width)
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        return x + self.net(x + self.time(temb))


class StructuredDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        n_blocks: int = 6,
        time_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.in_proj = nn.Linear(input_dim, hidden_size)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_size, time_dim, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.SiLU(), nn.Linear(hidden_size, input_dim))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h)
