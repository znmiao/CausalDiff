from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def windows_to_series(windows: np.ndarray) -> np.ndarray:
    if windows.ndim == 2:
        return windows.astype(np.float32)
    if windows.ndim != 3:
        raise ValueError(f"expected 2D or 3D time series, got shape {windows.shape}")
    return windows.reshape(-1, windows.shape[-1]).astype(np.float32)


def write_tcdf_csv(windows: np.ndarray, path: str | Path) -> None:
    arr = windows_to_series(windows)
    columns = [f"x{i}" for i in range(arr.shape[-1])]
    pd.DataFrame(arr, columns=columns).to_csv(path, index=False)


def run_tcdf_probe(
    windows: np.ndarray,
    tcdf_root: str | Path,
    max_lag: int,
    epochs: int = 1000,
    cuda: bool = False,
    seed: int = 1111,
) -> np.ndarray:
    tcdf_root = Path(tcdf_root)
    if not (tcdf_root / "runTCDF.py").exists():
        raise FileNotFoundError(f"runTCDF.py not found in {tcdf_root}")
    d = windows.shape[-1]
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "series.csv"
        write_tcdf_csv(windows, data_path)
        cmd = [
            "python",
            str(tcdf_root / "runTCDF.py"),
            "--data",
            str(data_path),
            "--epochs",
            str(epochs),
            "--kernel_size",
            str(max_lag + 1),
            "--dilation_coefficient",
            str(max_lag + 1),
            "--seed",
            str(seed),
        ]
        if cuda:
            cmd.append("--cuda")
        proc = subprocess.run(cmd, cwd=str(tcdf_root), text=True, capture_output=True, check=True)
    return parse_tcdf_output(proc.stdout, d, max_lag)


def parse_tcdf_output(text: str, n_features: int, max_lag: int) -> np.ndarray:
    graph = np.zeros((n_features, n_features, max_lag), dtype=np.float32)
    pattern = re.compile(r"x(?P<src>\d+)\s+causes\s+x(?P<tgt>\d+)\s+with a delay of\s+(?P<delay>\d+)\s+time steps")
    for m in pattern.finditer(text):
        src = int(m.group("src"))
        tgt = int(m.group("tgt"))
        delay = int(m.group("delay"))
        if 0 <= src < n_features and 0 <= tgt < n_features:
            lag = min(max(delay, 1), max_lag) - 1
            graph[src, tgt, lag] = 1.0
    return graph[None]
