import torch
from torch import nn
import torch.nn.functional as F


def open_uniform_knots(grid_min: float, grid_max: float, grid_size: int, degree: int) -> torch.Tensor:
    inner = torch.linspace(grid_min, grid_max, grid_size + 1)
    left = inner.new_full((degree + 1,), grid_min)
    right = inner.new_full((degree + 1,), grid_max)
    return torch.cat([left, inner[1:-1], right])


class EdgeSplineKAN(nn.Module):
    def __init__(
        self,
        n_features: int,
        max_lag: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.max_lag = max_lag
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        knots = open_uniform_knots(grid_range[0], grid_range[1], grid_size, spline_order)
        self.register_buffer("knots", knots)
        scale = 1.0 / max(self.n_basis, 1)
        self.coefficients = nn.Parameter(scale * torch.randn(n_features, n_features, max_lag, self.n_basis))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        degree = self.spline_order
        x = x.clamp(float(knots[0]) + 1e-6, float(knots[-1]) - 1e-6)
        basis = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).to(x.dtype)
        for p in range(1, degree + 1):
            left_num = x.unsqueeze(-1) - knots[: -(p + 1)]
            left_den = knots[p:-1] - knots[: -(p + 1)]
            right_num = knots[p + 1 :] - x.unsqueeze(-1)
            right_den = knots[p + 1 :] - knots[1:-p]
            left = torch.where(left_den.abs() > 0, left_num / left_den.clamp_min(1e-12), torch.zeros_like(left_num))
            right = torch.where(right_den.abs() > 0, right_num / right_den.clamp_min(1e-12), torch.zeros_like(right_num))
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis[..., : self.n_basis]

    def edge_values(self, lagged_values: torch.Tensor) -> torch.Tensor:
        b = self.basis(lagged_values)
        return torch.einsum("bilk,ijlk->bijl", b, self.coefficients)

    def step(self, lagged_values: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        edge = self.edge_values(lagged_values)
        return torch.sum(edge * causal, dim=(1, 3))

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
        horizon = residual.shape[1]
        for h in range(horizon):
            hist = torch.stack(seq, dim=1)
            t = hist.shape[1]
            lagged = torch.stack([hist[:, t - lag, :] for lag in range(1, self.max_lag + 1)], dim=2)
            seq.append(self.step(lagged, causal) + residual[:, h])
        return torch.stack(seq, dim=1)

    def smoothness_penalty(self) -> torch.Tensor:
        if self.coefficients.shape[-1] < 3:
            return self.coefficients.square().mean()
        second = self.coefficients[..., 2:] - 2 * self.coefficients[..., 1:-1] + self.coefficients[..., :-2]
        return second.square().mean()

    def l1_penalty(self) -> torch.Tensor:
        return self.coefficients.abs().mean()

    def active_functions(self, threshold: float = 1e-4) -> torch.Tensor:
        return (self.coefficients.abs().mean(dim=-1) > threshold).to(torch.float32)
