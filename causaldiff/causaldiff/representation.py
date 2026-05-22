from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackSpec:
    window_size: int
    max_lag: int
    n_features: int

    @property
    def prefix_dim(self) -> int:
        return self.max_lag * self.n_features

    @property
    def causal_dim(self) -> int:
        return self.n_features * self.n_features * self.max_lag

    @property
    def residual_dim(self) -> int:
        return (self.window_size - self.max_lag) * self.n_features

    @property
    def total_dim(self) -> int:
        return self.prefix_dim + self.causal_dim + self.residual_dim

    def pack(self, prefix: torch.Tensor, causal: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return torch.cat([prefix.flatten(1), causal.flatten(1), residual.flatten(1)], dim=1)

    def unpack(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = z.shape[0]
        p0 = self.prefix_dim
        c0 = p0 + self.causal_dim
        prefix = z[:, :p0].reshape(b, self.max_lag, self.n_features)
        causal = z[:, p0:c0].reshape(b, self.n_features, self.n_features, self.max_lag)
        residual = z[:, c0:].reshape(b, self.window_size - self.max_lag, self.n_features)
        return prefix, causal, residual


@dataclass(frozen=True)
class ShiftPackSpec:
    window_size: int
    max_lag: int
    n_features: int

    @property
    def causal_dim(self) -> int:
        return self.n_features * self.n_features * self.max_lag

    @property
    def residual_dim(self) -> int:
        return (self.window_size - self.max_lag) * self.n_features

    @property
    def total_dim(self) -> int:
        return self.causal_dim + self.residual_dim

    def pack(self, delta_causal: torch.Tensor, delta_residual: torch.Tensor) -> torch.Tensor:
        return torch.cat([delta_causal.flatten(1), delta_residual.flatten(1)], dim=1)

    def unpack(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = z.shape[0]
        delta_causal = z[:, :self.causal_dim].reshape(b, self.n_features, self.n_features, self.max_lag)
        delta_residual = z[:, self.causal_dim:].reshape(b, self.window_size - self.max_lag, self.n_features)
        return delta_causal, delta_residual
