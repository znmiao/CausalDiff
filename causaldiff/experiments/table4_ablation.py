from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


ABLATIONS = [
    "full_lagged_graph",
    "correlation_graph",
    "static_causal_graph",
    "linear_mechanism",
    "no_residual",
    "full",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window-size", type=int, default=48)
    p.add_argument("--lengths", nargs="+", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[2026])
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--quick", action="store_true")
    args, extra = p.parse_known_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    lengths = args.lengths or [args.window_size]
    for length in lengths:
        for seed in args.seeds:
            for ablation in ABLATIONS:
                out = root / f"T{length}_seed{seed}_{ablation}"
                cmd = [
                    "python", "causaldiff_repro/run.py", "train-normal",
                    "--data", args.data,
                    "--output", str(out),
                    "--window-size", str(length),
                    "--max-lag", str(args.max_lag),
                    "--ablation", ablation,
                    "--evaluate",
                    "--device", args.device,
                    "--seed", str(seed),
                ]
                if args.quick:
                    cmd += ["--quick-eval", "--predictor-epochs", "3", "--mechanism-epochs", "3", "--diffusion-epochs", "3"]
                cmd += extra
                subprocess.run(cmd, check=True)
                metrics = pd.read_json(out / "metrics.json", typ="series").to_dict()
                metrics.update({"variant": ablation, "window_size": length, "seed": seed})
                rows.append(metrics)
    raw = pd.DataFrame(rows)
    raw.to_csv(root / "table4_ablation_runs.csv", index=False)
    metric_cols = [c for c in raw.columns if c not in {"variant", "window_size", "seed"}]
    summary = raw.groupby("variant")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(root / "table4_ablation.csv", index=False)


if __name__ == "__main__":
    main()
