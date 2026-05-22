from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from .config import CausalDiffConfig
from .representation import PackSpec, ShiftPackSpec
from .utils import ensure_dir, save_json


def save_artifacts(
    out_dir: str | Path,
    cfg: CausalDiffConfig,
    spec: PackSpec,
    scaler,
    threshold: float,
    selected_percentile: float | None = None,
) -> None:
    out = ensure_dir(out_dir)
    save_json(cfg, out / "config.json")
    save_json({"window_size": spec.window_size, "max_lag": spec.max_lag, "n_features": spec.n_features}, out / "pack_spec.json")
    save_json({"causal_threshold": threshold, "selected_percentile": selected_percentile}, out / "causal_threshold.json")
    joblib.dump(scaler, out / "scaler.joblib")


def save_samples(samples: dict[str, torch.Tensor], path: str | Path) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in samples.items()}
    np.savez_compressed(path, **arrays)


def load_pack_spec(path: str | Path) -> PackSpec:
    import json

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return PackSpec(**d)


def load_shift_spec(path: str | Path) -> ShiftPackSpec:
    import json

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ShiftPackSpec(**d)
