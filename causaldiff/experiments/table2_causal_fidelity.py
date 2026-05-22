from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaldiff.evaluate import causal_metrics
from causaldiff.external_tcdf import run_tcdf_probe


def load_causal(path: str, key: str | None = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        if key is None:
            key = "causal" if "causal" in data else list(data.keys())[0]
        return data[key]
    if p.suffix == ".pt":
        data = torch.load(p, map_location="cpu")
        if isinstance(data, dict):
            key = key or ("test" if "test" in data else next(iter(data)))
            return data[key].numpy()
        return data.numpy()
    return np.load(p)


def summary_to_lagged(graph: np.ndarray, max_lag: int) -> np.ndarray:
    if graph.ndim == 2:
        return np.repeat(graph[None, :, :, None], max_lag, axis=-1)
    if graph.ndim == 3 and graph.shape[-1] != max_lag:
        return graph[None]
    if graph.ndim == 3:
        return graph[None]
    return graph


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--generated", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--ref-key", default=None)
    p.add_argument("--gen-key", default=None)
    p.add_argument("--tcdf-root", default=None)
    p.add_argument("--tcdf-epochs", type=int, default=1000)
    p.add_argument("--tcdf-cuda", action="store_true")
    p.add_argument("--tensor-only", action="store_true")
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--graph-type", choices=["summary", "lagged"], default="lagged")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_causal(args.reference, args.ref_key)
    rows = []
    for name, path in zip(args.names, args.generated):
        data = np.load(path) if Path(path).suffix == ".npz" else None
        if args.tcdf_root:
            if data is None or "x" not in data:
                raise ValueError("TCDF mode requires generated .npz files containing key 'x'")
            fake = run_tcdf_probe(data["x"].astype(np.float32), args.tcdf_root, ref.shape[-1] if ref.ndim >= 3 else args.max_lag, args.tcdf_epochs, args.tcdf_cuda)
        elif args.tensor_only:
            fake = load_causal(path, args.gen_key)
        else:
            raise ValueError("Use --tcdf-root for paper-style TCDF probing, or --tensor-only for precomputed causal tensors.")
        max_lag = fake.shape[-1] if fake.ndim == 4 else ref.shape[-1] if ref.ndim >= 3 else 1
        r = summary_to_lagged(ref, max_lag)
        f = summary_to_lagged(fake, max_lag)
        n = min(len(r), len(f))
        if len(r) == 1:
            r = np.repeat(r, n, axis=0)
        th = args.threshold
        if th is None:
            vals = r[r > 0]
            th = float(np.percentile(vals, 90)) if vals.size else 0.0
        row = {"method": name}
        if args.graph_type == "summary":
            r_edge = (r[:n].max(axis=-1) > th).astype(np.int32).reshape(-1)
            f_score = f[:n].max(axis=-1).reshape(-1)
            f_edge = (f_score > th).astype(np.int32)
            row.update({
                "f1": float(f1_score(r_edge, f_edge, zero_division=0)),
                "auprc": float(average_precision_score(r_edge, f_score)) if np.any(r_edge) else 0.0,
                "nshd": float(np.mean(r_edge != f_edge)),
            })
        else:
            row.update(causal_metrics(r[:n], f[:n], th))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "table2_causal_fidelity.csv", index=False)


if __name__ == "__main__":
    main()
