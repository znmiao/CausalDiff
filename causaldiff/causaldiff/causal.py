from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_lagged_causal_tensor(
    predictor: torch.nn.Module,
    windows: torch.Tensor,
    max_lag: int,
    device: torch.device,
) -> torch.Tensor:
    predictor.eval()
    x = windows.to(device)
    b, t_total, d = x.shape
    scores = x.new_zeros(b, d, d, max_lag)
    valid = t_total - max_lag
    for t in range(max_lag, t_total):
        history = x[:, t - max_lag : t].detach().clone().requires_grad_(True)
        target = x[:, t]
        pred = predictor(history)
        grads = []
        for j in range(d):
            predictor.zero_grad(set_to_none=True)
            if history.grad is not None:
                history.grad.zero_()
            err = (pred[:, j] - target[:, j]).square().sum()
            grad = torch.autograd.grad(err, history, retain_graph=True, create_graph=False)[0].abs()
            grad = torch.stack([grad[:, max_lag - lag, :] for lag in range(1, max_lag + 1)], dim=2)
            grads.append(grad)
        local = torch.stack(grads, dim=2)
        scores = scores + local
    return scores / max(valid, 1)


def extract_from_loader(
    predictor: torch.nn.Module,
    loader: DataLoader,
    max_lag: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for batch in loader:
        chunks.append(extract_lagged_causal_tensor(predictor, batch["x"], max_lag, device).cpu())
    return torch.cat(chunks, dim=0)


def direction_enhance(causal: torch.Tensor) -> torch.Tensor:
    out = causal.clone()
    d = causal.shape[1]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            out[:, i, j] = torch.relu(causal[:, i, j] - causal[:, j, i])
    return out


def sparsify(causal: torch.Tensor, threshold: float) -> torch.Tensor:
    enhanced = direction_enhance(causal)
    return torch.where(enhanced > threshold, enhanced, torch.zeros_like(enhanced))


def percentile_threshold(causal: torch.Tensor, percentile: float = 90.0) -> float:
    enhanced = direction_enhance(causal).detach().cpu().numpy()
    values = enhanced[enhanced > 0]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def select_threshold(
    causal: torch.Tensor,
    candidates: tuple[float, ...] = (80.0, 85.0, 90.0, 95.0),
    preferred: float = 90.0,
) -> tuple[float, float]:
    enhanced = direction_enhance(causal).detach()
    values = enhanced[enhanced > 0]
    if values.numel() == 0:
        return 0.0, candidates[0]
    best_percentile = min(candidates, key=lambda p: abs(p - preferred))
    best_threshold = float(np.percentile(values.cpu().numpy(), best_percentile))
    return best_threshold, best_percentile


def correlation_causal_tensor(windows: torch.Tensor, max_lag: int, eps: float = 1e-8) -> torch.Tensor:
    x = windows.float()
    b, t, d = x.shape
    out = x.new_zeros(b, d, d, max_lag)
    for lag in range(1, max_lag + 1):
        src = x[:, : t - lag]
        dst = x[:, lag:]
        src = src - src.mean(dim=1, keepdim=True)
        dst = dst - dst.mean(dim=1, keepdim=True)
        cov = torch.einsum("bti,btj->bij", src, dst) / max(t - lag, 1)
        std_src = torch.sqrt(src.square().mean(dim=1) + eps)
        std_dst = torch.sqrt(dst.square().mean(dim=1) + eps)
        out[:, :, :, lag - 1] = torch.abs(cov / (std_src.unsqueeze(2) * std_dst.unsqueeze(1) + eps))
    return out


def dense_lagged_tensor(windows: torch.Tensor, max_lag: int) -> torch.Tensor:
    b, _, d = windows.shape
    return torch.ones(b, d, d, max_lag, dtype=windows.dtype)


def static_tensor(causal: torch.Tensor) -> torch.Tensor:
    mean = causal.mean(dim=0, keepdim=True)
    return mean.repeat(causal.shape[0], 1, 1, 1)


def apply_causal_ablation(causal: torch.Tensor, windows: torch.Tensor, mode: str, reference: torch.Tensor | None = None) -> torch.Tensor:
    if mode in {"full", "linear_mechanism", "no_residual"}:
        return causal
    if mode == "full_lagged_graph":
        return dense_lagged_tensor(windows, causal.shape[-1]).to(causal.device)
    if mode == "correlation_graph":
        return correlation_causal_tensor(windows.to(causal.device), causal.shape[-1]).to(causal.device)
    if mode == "static_causal_graph":
        ref = causal if reference is None else reference.to(causal.device)
        return ref.mean(dim=0, keepdim=True).repeat(causal.shape[0], 1, 1, 1)
    raise ValueError(f"unknown ablation mode: {mode}")


def causal_edge_score(causal: torch.Tensor) -> torch.Tensor:
    return causal.max(dim=-1).values


def normalize_causal(causal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scale = causal.flatten(1).mean(dim=1).reshape(-1, 1, 1, 1).clamp_min(eps)
    return causal / scale
