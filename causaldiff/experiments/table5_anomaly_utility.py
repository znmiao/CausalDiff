from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causaldiff.anomaly import (
    anomaly_scores,
    build_mechanism_from_cfg,
    build_predictor_from_cfg,
    calibrate_with_generated,
    evaluate_threshold,
    load_scaled_windows,
)
from causaldiff.config import CausalDiffConfig
from causaldiff.representation import PackSpec
from causaldiff.utils import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--normal-dir", required=True)
    p.add_argument("--test-data", default=None)
    p.add_argument("--generated-anomalies", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--detector", choices=["oracle_scores", "internal"], default="oracle_scores")
    p.add_argument("--score-file", default=None)
    p.add_argument("--generated-score-file", default=None)
    p.add_argument("--label-column", default="-1")
    p.add_argument("--fewshot", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    normal_dir = Path(args.normal_dir)
    with open(normal_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = CausalDiffConfig(**json.load(f))
    with open(normal_dir / "pack_spec.json", "r", encoding="utf-8") as f:
        spec = PackSpec(**json.load(f))
    device = resolve_device(args.device or cfg.device)
    if args.detector == "oracle_scores":
        if args.score_file is None or args.generated_score_file is None:
            raise ValueError("OracleAD protocol requires --score-file and --generated-score-file.")
        score_pack = np.load(args.score_file)
        gen_pack = np.load(args.generated_score_file)
        scores = score_pack["test_scores"]
        labels = score_pack["test_labels"].astype(np.int32)
        normal_scores = score_pack["normal_val_scores"] if "normal_val_scores" in score_pack else scores[labels == 0]
        if "fewshot_scores" in score_pack:
            fewshot_scores = score_pack["fewshot_scores"]
        else:
            real_anom_scores = scores[labels > 0]
            fewshot_scores = real_anom_scores[: min(args.fewshot, len(real_anom_scores))]
        gen_scores = gen_pack["generated_scores"]
    else:
        if args.test_data is None or args.generated_anomalies is None:
            raise ValueError("internal detector requires --test-data and --generated-anomalies.")
        scaler = joblib.load(normal_dir / "scaler.joblib")
        label_column = int(args.label_column) if args.label_column.lstrip("-").isdigit() else args.label_column
        test_windows, labels = load_scaled_windows(args.test_data, scaler, spec.window_size, cfg.stride, label_column)
        if labels is None:
            raise ValueError("test labels are required for table 5 evaluation")
        causal_pack = torch.load(normal_dir / "causal_tensors.pt", map_location="cpu")
        reference_causal = causal_pack["train"]
        predictor = build_predictor_from_cfg(cfg, spec.n_features).to(device)
        predictor.load_state_dict(torch.load(normal_dir / "predictor.pt", map_location=device))
        mechanism = build_mechanism_from_cfg(cfg, spec.n_features).to(device)
        mechanism.load_state_dict(torch.load(normal_dir / "kan_mechanism.pt", map_location=device))
        scores = anomaly_scores(predictor, mechanism, test_windows, reference_causal, spec.max_lag, cfg.batch_size, device)
        normal_scores = scores[labels == 0]
        real_anom_scores = scores[labels > 0]
        fewshot_scores = real_anom_scores[: min(args.fewshot, len(real_anom_scores))]
        gen = np.load(args.generated_anomalies)["x"].astype(np.float32)
        gen_scores = anomaly_scores(predictor, mechanism, gen, reference_causal, spec.max_lag, cfg.batch_size, device)
    th = calibrate_with_generated(normal_scores, fewshot_scores, gen_scores)
    result = evaluate_threshold(scores, labels.astype(np.int32), th)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out / "table5_anomaly_utility.csv", index=False)


if __name__ == "__main__":
    main()
