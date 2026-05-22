from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaldiff.data import load_array, make_windows
from causaldiff.evaluate import generation_report
from causaldiff.utils import resolve_device


def load_real(path: str, window_size: int, stride: int, label_column=None, scaler=None) -> np.ndarray:
    values, _ = load_array(path, label_column)
    if scaler is not None:
        shape = values.shape
        values = scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    if values.ndim == 3:
        return values.astype(np.float32)
    windows, _ = make_windows(values, window_size, stride, None)
    return windows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--generated", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--normal-dir", default=None)
    p.add_argument("--window-size", type=int, default=48)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--label-column", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    label_column = int(args.label_column) if args.label_column and args.label_column.lstrip("-").isdigit() else args.label_column
    scaler = joblib.load(Path(args.normal_dir) / "scaler.joblib") if args.normal_dir else None
    real = load_real(args.real, args.window_size, args.stride, label_column, scaler)
    rows = []
    for name, path in zip(args.names, args.generated):
        gen = np.load(path)["x"].astype(np.float32)
        n = min(len(real), len(gen))
        row = {"method": name}
        row.update(generation_report(real[:n], gen[:n], device, quick=args.quick))
        rows.append(row)
        if args.plots:
            from causaldiff.plots import window_embedding_plots

            window_embedding_plots(real[:n], gen[:n], out / "plots", name)
    pd.DataFrame(rows).to_csv(out / "table1_generation_quality.csv", index=False)


if __name__ == "__main__":
    main()
