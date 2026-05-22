from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaldiff.anomaly import build_predictor_from_cfg
from causaldiff.causal import extract_from_loader, select_threshold, sparsify
from causaldiff.config import CausalDiffConfig
from causaldiff.data import WindowDataset
from causaldiff.evaluate import causal_metrics
from causaldiff.representation import PackSpec
from causaldiff.utils import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--normal-dir", required=True)
    p.add_argument("--generated", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference-key", default="test")
    p.add_argument("--scale-generated", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    normal_dir = Path(args.normal_dir)
    with open(normal_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = CausalDiffConfig(**json.load(f))
    with open(normal_dir / "pack_spec.json", "r", encoding="utf-8") as f:
        spec = PackSpec(**json.load(f))
    device = resolve_device(args.device or cfg.device)
    predictor = build_predictor_from_cfg(cfg, spec.n_features).to(device)
    predictor.load_state_dict(torch.load(normal_dir / "predictor.pt", map_location=device))
    reference = torch.load(normal_dir / "causal_tensors.pt", map_location="cpu")[args.reference_key]
    threshold, percentile = select_threshold(reference, cfg.threshold_candidates, cfg.causal_percentile)
    scaler = joblib.load(normal_dir / "scaler.joblib")
    rows = []
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, path in zip(args.names, args.generated):
        data = np.load(path)
        x = data["x"] if "x" in data else data[list(data.keys())[0]]
        x = x.astype(np.float32)
        if args.scale_generated:
            shape = x.shape
            x = scaler.transform(x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        if "causal" in data:
            causal = torch.tensor(data["causal"], dtype=torch.float32)
        else:
            causal = extract_from_loader(predictor, DataLoader(WindowDataset(x), batch_size=cfg.batch_size), spec.max_lag, device)
        n = min(len(reference), len(causal))
        r = sparsify(reference[:n], threshold).numpy()
        c = sparsify(causal[:n], threshold).numpy()
        row = {"method": name, "selected_percentile": percentile}
        row.update(causal_metrics(r, c, threshold))
        rows.append(row)
        np.savez_compressed(out / f"{name}_estimated_causal.npz", causal=causal.numpy())
    pd.DataFrame(rows).to_csv(out / "table3_estimated_causal_consistency.csv", index=False)


if __name__ == "__main__":
    main()
