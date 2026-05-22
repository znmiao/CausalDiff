from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from causaldiff.causal import apply_causal_ablation, extract_from_loader, select_threshold, sparsify
from causaldiff.config import CausalDiffConfig
from causaldiff.data import WindowDataset, build_window_splits, load_array, make_loader, make_windows_with_starts
from causaldiff.decompose import RepresentationDataset, build_representation, build_shift_dataset
from causaldiff.evaluate import causal_metrics, generation_report
from causaldiff.io import save_artifacts, save_samples
from causaldiff.models import (
    EdgeSplineKAN,
    GCADPredictor,
    GaussianDiffusion,
    LinearCausalMechanism,
    PackedResNetDenoiser,
    StructuredDenoiser,
    TemporalPredictor,
)
from causaldiff.representation import PackSpec, ShiftPackSpec
from causaldiff.training import (
    sample_anomalies,
    sample_normal,
    train_mechanism,
    train_predictor,
    train_representation_diffusion,
    train_shift_diffusion,
)
from causaldiff.utils import ensure_dir, resolve_device, save_json, set_seed


def cfg_from_args(args: argparse.Namespace) -> CausalDiffConfig:
    cfg = CausalDiffConfig(
        window_size=args.window_size,
        max_lag=args.max_lag,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stride=args.stride,
        batch_size=args.batch_size,
        predictor_type=args.predictor,
        denoiser_type=args.denoiser,
        gcad_blocks=args.gcad_blocks,
        gcad_ff_dim=args.gcad_ff_dim,
        denoiser_hidden=args.denoiser_hidden,
        denoiser_blocks=args.denoiser_blocks,
        dropout=args.dropout,
        ablation=args.ablation,
        lr=args.lr,
        predictor_epochs=args.predictor_epochs,
        mechanism_epochs=args.mechanism_epochs,
        diffusion_epochs=args.diffusion_epochs,
        patience=args.patience,
        diffusion_steps=args.diffusion_steps,
        ddim_steps=args.ddim_steps,
        causal_percentile=args.causal_percentile,
        seed=args.seed,
        device=args.device,
    )
    return cfg


def train_normal(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = ensure_dir(args.output)
    splits = build_window_splits(
        args.data,
        cfg.window_size,
        cfg.stride,
        args.train_ratio,
        args.val_ratio,
        parse_label_column(args.label_column),
    )
    n_features = splits.train.shape[-1]
    spec = PackSpec(cfg.window_size, cfg.max_lag, n_features)
    train_loader = make_loader(splits.train, splits.train_labels, cfg.batch_size, True)
    val_loader = make_loader(splits.val, splits.val_labels, cfg.batch_size, False)
    test_loader = make_loader(splits.test, splits.test_labels, cfg.batch_size, False)

    predictor = build_predictor(cfg, n_features, args)
    predictor = train_predictor(predictor, train_loader, val_loader, cfg, device, out / "predictor.pt")

    train_causal = extract_from_loader(predictor, train_loader, cfg.max_lag, device)
    val_causal = extract_from_loader(predictor, val_loader, cfg.max_lag, device)
    test_causal = extract_from_loader(predictor, test_loader, cfg.max_lag, device)
    train_causal = apply_causal_ablation(train_causal, torch.tensor(splits.train), cfg.ablation)
    val_causal = apply_causal_ablation(val_causal, torch.tensor(splits.val), cfg.ablation, train_causal)
    test_causal = apply_causal_ablation(test_causal, torch.tensor(splits.test), cfg.ablation, train_causal)
    threshold, selected_percentile = select_threshold(val_causal, cfg.threshold_candidates, cfg.causal_percentile)
    torch.save({"train": train_causal, "val": val_causal, "test": test_causal}, out / "causal_tensors.pt")

    mechanism = build_mechanism(cfg, n_features)
    mechanism = train_mechanism(
        mechanism,
        torch.tensor(splits.train),
        train_causal,
        torch.tensor(splits.val),
        val_causal,
        cfg,
        device,
        out / "kan_mechanism.pt",
    )

    use_residual = cfg.ablation != "no_residual"
    train_repr = build_representation(mechanism, torch.tensor(splits.train), train_causal, spec, device, cfg.batch_size, use_residual)
    val_repr = build_representation(mechanism, torch.tensor(splits.val), val_causal, spec, device, cfg.batch_size, use_residual)
    test_repr = build_representation(mechanism, torch.tensor(splits.test), test_causal, spec, device, cfg.batch_size, use_residual)
    torch.save({"train_z": train_repr.z, "val_z": val_repr.z, "test_z": test_repr.z}, out / "representations.pt")

    rep_train_loader = DataLoader(train_repr, batch_size=cfg.batch_size, shuffle=True)
    rep_val_loader = DataLoader(val_repr, batch_size=cfg.batch_size, shuffle=False)
    denoiser = build_denoiser(cfg, spec.total_dim)
    diffusion = GaussianDiffusion(cfg.diffusion_steps, cfg.beta_schedule, device=device)
    denoiser = train_representation_diffusion(
        denoiser, diffusion, rep_train_loader, rep_val_loader, mechanism, spec, cfg, device, out / "normal_diffusion.pt"
    )

    n_samples = args.samples if args.samples > 0 else len(splits.test)
    samples = sample_normal(denoiser, diffusion, mechanism, spec, n_samples, cfg, device)
    save_samples(samples, out / "generated_normal.npz")
    save_artifacts(out, cfg, spec, splits.scaler, threshold, selected_percentile)

    if args.evaluate:
        real = splits.test[:n_samples]
        fake = samples["x"].numpy()
        report = generation_report(real, fake, device, quick=args.quick_eval)
        if test_repr.causal.shape[0] >= n_samples:
            c_real = sparsify(test_repr.causal[:n_samples], threshold).numpy()
            c_fake = sparsify(samples["causal"], threshold).numpy()
            report.update(causal_metrics(c_real, c_fake, threshold))
        save_json(report, out / "metrics.json")


def train_shift(args: argparse.Namespace) -> None:
    normal_dir = Path(args.normal_dir)
    with open(normal_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = CausalDiffConfig(**json.load(f))
    set_seed(args.seed if args.seed is not None else cfg.seed)
    device = resolve_device(args.device or cfg.device)
    out = ensure_dir(args.output)
    with open(normal_dir / "pack_spec.json", "r", encoding="utf-8") as f:
        spec = PackSpec(**json.load(f))
    shift_spec = ShiftPackSpec(spec.window_size, spec.max_lag, spec.n_features)
    scaler = joblib.load(normal_dir / "scaler.joblib")

    normal_windows, normal_starts = _load_scaled_windows(args.normal_data, scaler, spec.window_size, cfg.stride, parse_label_column(args.normal_label_column), normal_only=True)
    anomaly_windows, anomaly_starts = _load_scaled_windows(args.anomaly_data, scaler, spec.window_size, cfg.stride, parse_label_column(args.anomaly_label_column), normal_only=False)
    normal_loader = DataLoader(WindowDataset(normal_windows), batch_size=cfg.batch_size, shuffle=False)
    anomaly_loader = DataLoader(WindowDataset(anomaly_windows), batch_size=cfg.batch_size, shuffle=False)

    predictor = build_predictor(cfg, spec.n_features, args).to(device)
    predictor.load_state_dict(torch.load(normal_dir / "predictor.pt", map_location=device))
    mechanism = build_mechanism(cfg, spec.n_features).to(device)
    mechanism.load_state_dict(torch.load(normal_dir / "kan_mechanism.pt", map_location=device))

    normal_causal = extract_from_loader(predictor, normal_loader, spec.max_lag, device)
    anomaly_causal = extract_from_loader(predictor, anomaly_loader, spec.max_lag, device)
    use_residual = cfg.ablation != "no_residual"
    normal_repr = build_representation(mechanism, torch.tensor(normal_windows), normal_causal, spec, device, cfg.batch_size, use_residual)
    anomaly_repr = build_representation(mechanism, torch.tensor(anomaly_windows), anomaly_causal, spec, device, cfg.batch_size, use_residual)
    shift = build_shift_dataset(
        normal_repr.causal,
        normal_repr.residual,
        anomaly_repr.causal,
        anomaly_repr.residual,
        shift_spec,
        torch.tensor(normal_starts),
        torch.tensor(anomaly_starts),
    )
    n_val = max(1, int(0.1 * len(shift)))
    n_train = len(shift) - n_val
    train_ds, val_ds = random_split(shift, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    denoiser = build_denoiser(cfg, shift_spec.total_dim, args.denoiser_hidden, args.denoiser_blocks, args.dropout)
    diffusion = GaussianDiffusion(cfg.diffusion_steps, cfg.beta_schedule, device=device)
    denoiser = train_shift_diffusion(denoiser, diffusion, train_loader, val_loader, cfg, device, out / "shift_diffusion.pt")

    normal_denoiser = build_denoiser(cfg, spec.total_dim).to(device)
    normal_denoiser.load_state_dict(torch.load(normal_dir / "normal_diffusion.pt", map_location=device))
    normal_diffusion = GaussianDiffusion(cfg.diffusion_steps, cfg.beta_schedule, device=device)
    base = sample_normal(normal_denoiser, normal_diffusion, mechanism, spec, args.samples, cfg, device)
    anomalies = sample_anomalies(base, denoiser, diffusion, mechanism, shift_spec, cfg, device, args.lambda_c, args.lambda_u)
    save_samples(anomalies, out / "generated_anomalies.npz")
    save_json({"window_size": shift_spec.window_size, "max_lag": shift_spec.max_lag, "n_features": shift_spec.n_features}, out / "shift_pack_spec.json")


def _load_scaled_windows(
    path: str,
    scaler,
    window_size: int,
    stride: int,
    label_column: str | int | None,
    normal_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values, labels = load_array(path, label_column)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    values = scaler.transform(flat).reshape(shape).astype(np.float32)
    windows, window_labels, starts = make_windows_with_starts(values, window_size, stride, labels, 0)
    if window_labels is not None:
        mask = window_labels == 0 if normal_only else window_labels > 0
        windows = windows[mask]
        starts = starts[mask]
    return windows, starts


def build_predictor(cfg: CausalDiffConfig, n_features: int, args: argparse.Namespace):
    if cfg.predictor_type == "gcad":
        return GCADPredictor(
            n_features=n_features,
            seq_len=cfg.max_lag,
            pred_len=1,
            n_blocks=cfg.gcad_blocks,
            ff_dim=cfg.gcad_ff_dim,
            dropout=cfg.dropout,
        )
    return TemporalPredictor(n_features, cfg.predictor_hidden, cfg.predictor_kernel)


def build_mechanism(cfg: CausalDiffConfig, n_features: int):
    if cfg.ablation == "linear_mechanism":
        return LinearCausalMechanism(n_features, cfg.max_lag)
    return EdgeSplineKAN(
        n_features,
        cfg.max_lag,
        cfg.kan_grid_size,
        cfg.kan_spline_order,
        (cfg.kan_grid_min, cfg.kan_grid_max),
    )


def build_denoiser(
    cfg: CausalDiffConfig,
    input_dim: int,
    hidden: int | None = None,
    blocks: int | None = None,
    dropout: float | None = None,
):
    hidden = cfg.denoiser_hidden if hidden is None else hidden
    blocks = cfg.denoiser_blocks if blocks is None else blocks
    dropout = cfg.dropout if dropout is None else dropout
    if cfg.denoiser_type == "mlp":
        return StructuredDenoiser(input_dim, hidden, blocks, dropout=dropout)
    return PackedResNetDenoiser(input_dim, hidden, blocks, dropout=dropout)


def parse_label_column(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CausalDiff reproduction")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("train-normal")
    n.add_argument("--data", required=True)
    n.add_argument("--output", required=True)
    n.add_argument("--label-column", default=None)
    n.add_argument("--window-size", type=int, default=48)
    n.add_argument("--max-lag", type=int, default=8)
    n.add_argument("--stride", type=int, default=1)
    n.add_argument("--train-ratio", type=float, default=0.7)
    n.add_argument("--val-ratio", type=float, default=0.1)
    n.add_argument("--batch-size", type=int, default=64)
    n.add_argument("--predictor", choices=["tcn", "gcad"], default="tcn")
    n.add_argument("--denoiser", choices=["resnet", "mlp"], default="resnet")
    n.add_argument("--ablation", choices=["full", "full_lagged_graph", "correlation_graph", "static_causal_graph", "linear_mechanism", "no_residual"], default="full")
    n.add_argument("--lr", type=float, default=1e-4)
    n.add_argument("--predictor-epochs", type=int, default=300)
    n.add_argument("--mechanism-epochs", type=int, default=300)
    n.add_argument("--diffusion-epochs", type=int, default=300)
    n.add_argument("--patience", type=int, default=30)
    n.add_argument("--diffusion-steps", type=int, default=100)
    n.add_argument("--ddim-steps", type=int, default=50)
    n.add_argument("--causal-percentile", type=float, default=90.0)
    n.add_argument("--samples", type=int, default=0)
    n.add_argument("--denoiser-hidden", type=int, default=512)
    n.add_argument("--denoiser-blocks", type=int, default=6)
    n.add_argument("--gcad-blocks", type=int, default=3)
    n.add_argument("--gcad-ff-dim", type=int, default=1024)
    n.add_argument("--dropout", type=float, default=0.0)
    n.add_argument("--evaluate", action="store_true")
    n.add_argument("--quick-eval", action="store_true")
    n.add_argument("--seed", type=int, default=2026)
    n.add_argument("--device", default="cuda")
    n.set_defaults(func=train_normal)

    s = sub.add_parser("train-shift")
    s.add_argument("--normal-dir", required=True)
    s.add_argument("--normal-data", required=True)
    s.add_argument("--anomaly-data", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--normal-label-column", default=None)
    s.add_argument("--anomaly-label-column", default=None)
    s.add_argument("--samples", type=int, default=256)
    s.add_argument("--lambda-c", type=float, default=1.0)
    s.add_argument("--lambda-u", type=float, default=1.0)
    s.add_argument("--denoiser-hidden", type=int, default=512)
    s.add_argument("--denoiser-blocks", type=int, default=6)
    s.add_argument("--gcad-blocks", type=int, default=3)
    s.add_argument("--gcad-ff-dim", type=int, default=1024)
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--device", default=None)
    s.set_defaults(func=train_shift)
    return p


if __name__ == "__main__":
    parser = build_parser()
    parsed = parser.parse_args()
    parsed.func(parsed)
