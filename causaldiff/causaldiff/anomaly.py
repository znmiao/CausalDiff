from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader

from .causal import extract_from_loader
from .data import WindowDataset, load_array, make_windows
from .models import EdgeSplineKAN, GCADPredictor, LinearCausalMechanism, TemporalPredictor
from .representation import PackSpec


def load_scaled_windows(path: str, scaler, window_size: int, stride: int, label_column=None) -> tuple[np.ndarray, np.ndarray | None]:
    values, labels = load_array(path, label_column)
    shape = values.shape
    values = scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    return make_windows(values, window_size, stride, labels)


def build_predictor_from_cfg(cfg, n_features: int):
    if cfg.predictor_type == "gcad":
        return GCADPredictor(n_features, cfg.max_lag, 1, cfg.gcad_blocks, cfg.gcad_ff_dim, cfg.dropout)
    return TemporalPredictor(n_features, cfg.predictor_hidden, cfg.predictor_kernel)


def build_mechanism_from_cfg(cfg, n_features: int):
    if cfg.ablation == "linear_mechanism":
        return LinearCausalMechanism(n_features, cfg.max_lag)
    return EdgeSplineKAN(n_features, cfg.max_lag, cfg.kan_grid_size, cfg.kan_spline_order, (cfg.kan_grid_min, cfg.kan_grid_max))


def anomaly_scores(
    predictor,
    mechanism,
    windows: np.ndarray,
    reference_causal: torch.Tensor,
    max_lag: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(WindowDataset(windows), batch_size=batch_size, shuffle=False)
    causal = extract_from_loader(predictor, loader, max_lag, device).to(device)
    ref = reference_causal.to(device).mean(dim=0, keepdim=True)
    scores = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.tensor(windows[start : start + batch_size], device=device)
            c = causal[start : start + batch_size]
            pred_err = torch.mean((mechanism.deterministic(x, c) - x[:, max_lag:]).abs(), dim=(1, 2))
            causal_err = torch.mean(torch.abs(c - ref) / (ref.abs() + 1e-4), dim=(1, 2, 3))
            scores.append((pred_err + causal_err).cpu())
    return torch.cat(scores).numpy()


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1))
    if idx >= len(thresholds):
        return float(scores.max() + 1e-6), float(f1[idx])
    return float(thresholds[idx]), float(f1[idx])


def calibrate_with_generated(
    normal_scores: np.ndarray,
    fewshot_scores: np.ndarray,
    generated_scores: np.ndarray,
) -> float:
    scores = np.concatenate([normal_scores, fewshot_scores, generated_scores])
    labels = np.concatenate([
        np.zeros_like(normal_scores, dtype=np.int32),
        np.ones_like(fewshot_scores, dtype=np.int32),
        np.ones_like(generated_scores, dtype=np.int32),
    ])
    th, _ = best_threshold(scores, labels)
    return th


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int32)
    return {
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "aupr": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
    }
