from dataclasses import dataclass


@dataclass
class CausalDiffConfig:
    window_size: int = 48
    max_lag: int = 8
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    stride: int = 1
    batch_size: int = 64
    predictor_type: str = "tcn"
    predictor_hidden: int = 128
    predictor_kernel: int = 3
    gcad_blocks: int = 3
    gcad_ff_dim: int = 1024
    denoiser_type: str = "resnet"
    denoiser_hidden: int = 512
    denoiser_blocks: int = 6
    dropout: float = 0.0
    ablation: str = "full"
    predictor_epochs: int = 300
    mechanism_epochs: int = 300
    diffusion_epochs: int = 300
    patience: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    diffusion_steps: int = 100
    ddim_steps: int = 50
    beta_schedule: str = "linear"
    kan_grid_size: int = 8
    kan_spline_order: int = 3
    kan_grid_min: float = -3.0
    kan_grid_max: float = 3.0
    lambda_p: float = 1.0
    lambda_c: float = 0.5
    lambda_u: float = 0.2
    lambda_roll: float = 0.1
    lambda_kan_smooth: float = 1e-4
    lambda_kan_l1: float = 1e-5
    causal_percentile: float = 90.0
    threshold_candidates: tuple[float, ...] = (80.0, 85.0, 90.0, 95.0)
    seed: int = 2026
    device: str = "cuda"

    @property
    def rollout_size(self) -> int:
        return self.window_size - self.max_lag
