from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .models.kan import EdgeSplineKAN
from .representation import PackSpec, ShiftPackSpec


class RepresentationDataset(Dataset):
    def __init__(self, windows: torch.Tensor, causal: torch.Tensor, residual: torch.Tensor, spec: PackSpec) -> None:
        self.windows = windows.float()
        self.causal = causal.float()
        self.residual = residual.float()
        self.prefix = self.windows[:, : spec.max_lag]
        self.z = spec.pack(self.prefix, self.causal, self.residual).float()

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.windows[idx], "z": self.z[idx]}


class ShiftDataset(Dataset):
    def __init__(self, delta_causal: torch.Tensor, delta_residual: torch.Tensor, spec: ShiftPackSpec) -> None:
        self.delta_causal = delta_causal.float()
        self.delta_residual = delta_residual.float()
        self.z = spec.pack(self.delta_causal, self.delta_residual).float()

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"z": self.z[idx]}


@torch.no_grad()
def build_representation(
    mechanism: EdgeSplineKAN,
    windows: torch.Tensor,
    causal: torch.Tensor,
    spec: PackSpec,
    device: torch.device,
    batch_size: int = 64,
    use_residual: bool = True,
) -> RepresentationDataset:
    mechanism.eval()
    residuals = []
    for start in range(0, windows.shape[0], batch_size):
        x = windows[start : start + batch_size].to(device)
        c = causal[start : start + batch_size].to(device)
        if use_residual:
            residuals.append(mechanism.residual(x, c).cpu())
        else:
            residuals.append(torch.zeros(x.shape[0], spec.window_size - spec.max_lag, spec.n_features).cpu())
    residual = torch.cat(residuals, dim=0)
    return RepresentationDataset(windows.cpu(), causal.cpu(), residual, spec)


def build_shift_dataset(
    normal_causal: torch.Tensor,
    normal_residual: torch.Tensor,
    anomaly_causal: torch.Tensor,
    anomaly_residual: torch.Tensor,
    spec: ShiftPackSpec,
    normal_starts: torch.Tensor | None = None,
    anomaly_starts: torch.Tensor | None = None,
) -> ShiftDataset:
    n = anomaly_causal.shape[0]
    if normal_starts is not None and anomaly_starts is not None and len(normal_starts) > 0:
        normal_starts = normal_starts.to(torch.long)
        anomaly_starts = anomaly_starts.to(torch.long)
        idx = torch.searchsorted(normal_starts, anomaly_starts).clamp(max=len(normal_starts) - 1)
        left = (idx - 1).clamp(min=0)
        choose_left = (anomaly_starts - normal_starts[left]).abs() <= (normal_starts[idx] - anomaly_starts).abs()
        idx = torch.where(choose_left, left, idx)
    else:
        idx = torch.arange(n) % normal_causal.shape[0]
    delta_causal = anomaly_causal - normal_causal[idx]
    delta_residual = anomaly_residual - normal_residual[idx]
    return ShiftDataset(delta_causal, delta_residual, spec)
