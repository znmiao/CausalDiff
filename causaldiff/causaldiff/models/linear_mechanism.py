import torch
from torch import nn


class LinearCausalMechanism(nn.Module):
    def __init__(self, n_features: int, max_lag: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.max_lag = max_lag
        self.weight = nn.Parameter(torch.randn(n_features, n_features, max_lag) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, n_features, max_lag))

    def edge_values(self, lagged_values: torch.Tensor) -> torch.Tensor:
        return lagged_values.unsqueeze(2) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

    def step(self, lagged_values: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.edge_values(lagged_values) * causal, dim=(1, 3))

    def deterministic(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        outs = []
        for t in range(self.max_lag, x.shape[1]):
            lagged = torch.stack([x[:, t - lag, :] for lag in range(1, self.max_lag + 1)], dim=2)
            outs.append(self.step(lagged, causal))
        return torch.stack(outs, dim=1)

    def residual(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        return x[:, self.max_lag :] - self.deterministic(x, causal)

    def rollout(self, prefix: torch.Tensor, causal: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        seq = [prefix[:, i] for i in range(prefix.shape[1])]
        for h in range(residual.shape[1]):
            hist = torch.stack(seq, dim=1)
            t = hist.shape[1]
            lagged = torch.stack([hist[:, t - lag, :] for lag in range(1, self.max_lag + 1)], dim=2)
            seq.append(self.step(lagged, causal) + residual[:, h])
        return torch.stack(seq, dim=1)

    def smoothness_penalty(self) -> torch.Tensor:
        return self.weight.square().mean()

    def l1_penalty(self) -> torch.Tensor:
        return self.weight.abs().mean()
