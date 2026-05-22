import torch
import torch.nn.functional as F
from torch import nn


class RevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise ValueError(mode)


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, n_features: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_time = nn.BatchNorm1d(seq_len * n_features)
        self.time = nn.Linear(seq_len, seq_len)
        self.drop_time = nn.Dropout(dropout)
        self.norm_feat = nn.BatchNorm1d(seq_len * n_features)
        self.feat_in = nn.Linear(n_features, ff_dim)
        self.feat_out = nn.Linear(ff_dim, n_features)
        self.drop_feat = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_time(torch.flatten(x, 1)).reshape_as(x)
        h = self.time(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.drop_time(F.relu(h))
        h = self.norm_feat(torch.flatten(x, 1)).reshape_as(x)
        h = self.feat_out(self.drop_feat(F.relu(self.feat_in(h))))
        return x + self.drop_feat(h)


class GCADPredictor(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int = 1,
        n_blocks: int = 3,
        ff_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.rev_norm = RevIN(n_features)
        self.blocks = nn.ModuleList([MixerBlock(seq_len, n_features, ff_dim, dropout) for _ in range(n_blocks)])
        self.proj = nn.Linear(seq_len, pred_len)

    def forward_sequence(self, history: torch.Tensor) -> torch.Tensor:
        x = self.rev_norm(history, "norm")
        for block in self.blocks:
            x = block(x)
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return self.rev_norm(x, "denorm")

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(history)[:, -1]
