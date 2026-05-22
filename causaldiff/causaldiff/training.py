from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CausalDiffConfig
from .models.diffusion import GaussianDiffusion
from .models.kan import EdgeSplineKAN
from .representation import PackSpec, ShiftPackSpec
from .utils import EarlyStopping, RunningMean


def train_predictor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CausalDiffConfig,
    device: torch.device,
    out_path: Path,
) -> nn.Module:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.patience)
    best = None
    for _ in range(cfg.predictor_epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            loss = _predictor_loss(model, x, cfg.max_lag)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate_predictor_loss(model, val_loader, cfg.max_lag, device)
        if val < stopper.best:
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best, out_path)
        if stopper.step(val):
            break
    if best is None:
        best = torch.load(out_path, map_location="cpu")
    model.load_state_dict(best)
    return model


def _predictor_loss(model: nn.Module, x: torch.Tensor, max_lag: int) -> torch.Tensor:
    losses = []
    for t in range(max_lag, x.shape[1]):
        pred = model(x[:, t - max_lag : t])
        losses.append(F.mse_loss(pred, x[:, t]))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate_predictor_loss(model: nn.Module, loader: DataLoader, max_lag: int, device: torch.device) -> float:
    model.eval()
    meter = RunningMean()
    for batch in loader:
        x = batch["x"].to(device)
        loss = _predictor_loss(model, x, max_lag)
        meter.update(float(loss), x.shape[0])
    return meter.value


def train_mechanism(
    mechanism: EdgeSplineKAN,
    train_windows: torch.Tensor,
    train_causal: torch.Tensor,
    val_windows: torch.Tensor,
    val_causal: torch.Tensor,
    cfg: CausalDiffConfig,
    device: torch.device,
    out_path: Path,
) -> EdgeSplineKAN:
    mechanism.to(device)
    opt = torch.optim.AdamW(mechanism.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.patience)
    best = None
    for _ in range(cfg.mechanism_epochs):
        mechanism.train()
        order = torch.randperm(train_windows.shape[0])
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            x = train_windows[idx].to(device)
            c = train_causal[idx].to(device)
            det = mechanism.deterministic(x, c)
            loss = F.mse_loss(det, x[:, cfg.max_lag :])
            loss = loss + cfg.lambda_kan_smooth * mechanism.smoothness_penalty()
            loss = loss + cfg.lambda_kan_l1 * mechanism.l1_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate_mechanism_loss(mechanism, val_windows, val_causal, cfg, device)
        if val < stopper.best:
            best = {k: v.detach().cpu() for k, v in mechanism.state_dict().items()}
            torch.save(best, out_path)
        if stopper.step(val):
            break
    if best is None:
        best = torch.load(out_path, map_location="cpu")
    mechanism.load_state_dict(best)
    return mechanism


@torch.no_grad()
def evaluate_mechanism_loss(
    mechanism: EdgeSplineKAN,
    windows: torch.Tensor,
    causal: torch.Tensor,
    cfg: CausalDiffConfig,
    device: torch.device,
) -> float:
    mechanism.eval()
    meter = RunningMean()
    for start in range(0, windows.shape[0], cfg.batch_size):
        x = windows[start : start + cfg.batch_size].to(device)
        c = causal[start : start + cfg.batch_size].to(device)
        loss = F.mse_loss(mechanism.deterministic(x, c), x[:, cfg.max_lag :])
        meter.update(float(loss), x.shape[0])
    return meter.value


def train_representation_diffusion(
    denoiser: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mechanism: EdgeSplineKAN,
    spec: PackSpec,
    cfg: CausalDiffConfig,
    device: torch.device,
    out_path: Path,
) -> nn.Module:
    return _train_structured_diffusion(
        denoiser, diffusion, train_loader, val_loader, mechanism, spec, cfg, device, out_path, is_shift=False
    )


def train_shift_diffusion(
    denoiser: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: CausalDiffConfig,
    device: torch.device,
    out_path: Path,
) -> nn.Module:
    return _train_structured_diffusion(
        denoiser, diffusion, train_loader, val_loader, None, None, cfg, device, out_path, is_shift=True
    )


def _train_structured_diffusion(
    denoiser: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mechanism: EdgeSplineKAN | None,
    spec: PackSpec | None,
    cfg: CausalDiffConfig,
    device: torch.device,
    out_path: Path,
    is_shift: bool,
) -> nn.Module:
    denoiser.to(device)
    if mechanism is not None:
        mechanism.to(device).eval()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopping(cfg.patience)
    best = None
    epochs = cfg.diffusion_epochs
    for _ in range(epochs):
        denoiser.train()
        for batch in train_loader:
            z0 = batch["z"].to(device)
            t = torch.randint(0, diffusion.timesteps, (z0.shape[0],), device=device, dtype=torch.long)
            zt = diffusion.q_sample(z0, t)
            zhat = denoiser(zt, t)
            if is_shift:
                loss = F.l1_loss(zhat, z0)
            else:
                x = batch["x"].to(device)
                loss = representation_loss(zhat, z0, x, mechanism, spec, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
        val = evaluate_diffusion_loss(denoiser, diffusion, val_loader, mechanism, spec, cfg, device, is_shift)
        if val < stopper.best:
            best = {k: v.detach().cpu() for k, v in denoiser.state_dict().items()}
            torch.save(best, out_path)
        if stopper.step(val):
            break
    if best is None:
        best = torch.load(out_path, map_location="cpu")
    denoiser.load_state_dict(best)
    return denoiser


def representation_loss(
    zhat: torch.Tensor,
    z0: torch.Tensor,
    x: torch.Tensor,
    mechanism: EdgeSplineKAN,
    spec: PackSpec,
    cfg: CausalDiffConfig,
) -> torch.Tensor:
    p, c, u = spec.unpack(z0)
    ph, ch, uh = spec.unpack(zhat)
    loss = cfg.lambda_p * F.l1_loss(ph, p)
    loss = loss + cfg.lambda_c * F.l1_loss(ch, c)
    if cfg.ablation != "no_residual":
        loss = loss + cfg.lambda_u * F.l1_loss(uh, u)
    else:
        uh = torch.zeros_like(uh)
    rollout = mechanism.rollout(ph, torch.relu(ch), uh)
    return loss + cfg.lambda_roll * F.mse_loss(rollout, x)


@torch.no_grad()
def evaluate_diffusion_loss(
    denoiser: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    mechanism: EdgeSplineKAN | None,
    spec: PackSpec | None,
    cfg: CausalDiffConfig,
    device: torch.device,
    is_shift: bool,
) -> float:
    denoiser.eval()
    meter = RunningMean()
    for batch in loader:
        z0 = batch["z"].to(device)
        t = torch.randint(0, diffusion.timesteps, (z0.shape[0],), device=device, dtype=torch.long)
        zhat = denoiser(diffusion.q_sample(z0, t), t)
        if is_shift:
            loss = F.l1_loss(zhat, z0)
        else:
            loss = representation_loss(zhat, z0, batch["x"].to(device), mechanism, spec, cfg)
        meter.update(float(loss), z0.shape[0])
    return meter.value


@torch.no_grad()
def sample_normal(
    denoiser: nn.Module,
    diffusion: GaussianDiffusion,
    mechanism: EdgeSplineKAN,
    spec: PackSpec,
    n_samples: int,
    cfg: CausalDiffConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    denoiser.to(device).eval()
    mechanism.to(device).eval()
    z = diffusion.ddim_sample(denoiser, (n_samples, spec.total_dim), steps=cfg.ddim_steps)
    prefix, causal, residual = spec.unpack(z)
    causal = torch.relu(causal)
    if cfg.ablation == "no_residual":
        residual = torch.zeros_like(residual)
    x = mechanism.rollout(prefix, causal, residual)
    return {"x": x.cpu(), "prefix": prefix.cpu(), "causal": causal.cpu(), "residual": residual.cpu(), "z": z.cpu()}


@torch.no_grad()
def sample_anomalies(
    normal: dict[str, torch.Tensor],
    shift_denoiser: nn.Module,
    shift_diffusion: GaussianDiffusion,
    mechanism: EdgeSplineKAN,
    shift_spec: ShiftPackSpec,
    cfg: CausalDiffConfig,
    device: torch.device,
    lambda_c: float = 1.0,
    lambda_u: float = 1.0,
) -> dict[str, torch.Tensor]:
    shift_denoiser.to(device).eval()
    mechanism.to(device).eval()
    n = normal["x"].shape[0]
    z = shift_diffusion.ddim_sample(shift_denoiser, (n, shift_spec.total_dim), steps=cfg.ddim_steps)
    dc, du = shift_spec.unpack(z)
    prefix = normal["prefix"].to(device)
    causal = torch.relu(normal["causal"].to(device) + lambda_c * dc)
    residual = normal["residual"].to(device) + lambda_u * du
    x = mechanism.rollout(prefix, causal, residual)
    return {"x": x.cpu(), "prefix": prefix.cpu(), "causal": causal.cpu(), "residual": residual.cpu(), "delta": z.cpu()}
