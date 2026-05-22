from __future__ import annotations

from typing import Optional

import torch


class GaussianDiffusion:
    def __init__(
        self,
        timesteps: int = 100,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)
        self.betas = self._schedule(schedule, timesteps, beta_start, beta_end).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _schedule(self, name: str, steps: int, start: float, end: float) -> torch.Tensor:
        if name == "cosine":
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            ac = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            ac = ac / ac[0]
            return torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.999)
        if name == "quadratic":
            return torch.linspace(start**0.5, end**0.5, steps) ** 2
        return torch.linspace(start, end, steps)

    def extract(self, values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values.gather(0, t).reshape(t.shape[0], *((1,) * (x.ndim - 1))).to(x.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        return self.extract(self.sqrt_alphas_cumprod, t, x0) * x0 + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def p_sample(self, model: torch.nn.Module, xt: torch.Tensor, t: torch.Tensor, step: int) -> torch.Tensor:
        x0 = model(xt, t).clamp(-20.0, 20.0)
        mean = self.extract(self.posterior_mean_coef1, t, xt) * x0 + self.extract(self.posterior_mean_coef2, t, xt) * xt
        if step == 0:
            return mean
        var = self.extract(self.posterior_variance, t, xt).clamp_min(1e-20)
        return mean + torch.sqrt(var) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, model: torch.nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
        xt = torch.randn(shape, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            xt = self.p_sample(model, xt, t, step)
        return xt

    @torch.no_grad()
    def ddim_sample(self, model: torch.nn.Module, shape: tuple[int, ...], steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        xt = torch.randn(shape, device=self.device)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        for i, step in enumerate(schedule):
            t = torch.full((shape[0],), int(step.item()), device=self.device, dtype=torch.long)
            x0 = model(xt, t).clamp(-20.0, 20.0)
            if i == len(schedule) - 1:
                xt = x0
                continue
            next_t = int(schedule[i + 1].item())
            a_t = self.alphas_cumprod[step]
            a_next = self.alphas_cumprod[next_t]
            eps = (xt - torch.sqrt(a_t) * x0) / torch.sqrt(1.0 - a_t)
            sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next)).clamp_min(0.0)
            c = torch.sqrt((1 - a_next - sigma**2).clamp_min(0.0))
            xt = torch.sqrt(a_next) * x0 + c * eps
            if eta > 0:
                xt = xt + sigma * torch.randn_like(xt)
        return xt
