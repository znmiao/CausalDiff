import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def correlation_score(real: np.ndarray, fake: np.ndarray) -> float:
    real_flat = real.reshape(-1, real.shape[-1])
    fake_flat = fake.reshape(-1, fake.shape[-1])
    c_real = np.nan_to_num(np.corrcoef(real_flat, rowvar=False))
    c_fake = np.nan_to_num(np.corrcoef(fake_flat, rowvar=False))
    return float(np.mean(np.abs(c_real - c_fake)))


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1]).squeeze(-1)


class GRUPredictor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128) -> None:
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1])


def discriminative_score(
    real: np.ndarray,
    fake: np.ndarray,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 128,
) -> float:
    n = min(len(real), len(fake))
    x = np.concatenate([real[:n], fake[:n]], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0).astype(np.float32)
    order = np.random.permutation(len(x))
    split = int(0.8 * len(x))
    train_idx, test_idx = order[:split], order[split:]
    train = DataLoader(TensorDataset(torch.tensor(x[train_idx]), torch.tensor(y[train_idx])), batch_size=batch_size, shuffle=True)
    test_x = torch.tensor(x[test_idx], device=device)
    test_y = torch.tensor(y[test_idx], device=device)
    model = GRUClassifier(real.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(test_x)) > 0.5).float()
    acc = (pred == test_y).float().mean().item()
    return float(abs(acc - 0.5))


def predictive_score(
    train_fake: np.ndarray,
    test_real: np.ndarray,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 128,
) -> dict[str, float]:
    train_x = torch.tensor(train_fake[:, :-1], dtype=torch.float32)
    train_y = torch.tensor(train_fake[:, -1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = GRUPredictor(train_fake.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    x = torch.tensor(test_real[:, :-1], dtype=torch.float32, device=device)
    y = torch.tensor(test_real[:, -1], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mse = nn.functional.mse_loss(pred, y).item()
        mae = nn.functional.l1_loss(pred, y).item()
    return {"mse": float(mse), "mae": float(mae)}


def binarize_causal(causal: np.ndarray, threshold: float) -> np.ndarray:
    return (causal > threshold).astype(np.int32)


def causal_metrics(real: np.ndarray, fake: np.ndarray, threshold: float) -> dict[str, float]:
    real_b = binarize_causal(real, threshold)
    fake_b = binarize_causal(fake, threshold)
    r = real_b.reshape(-1)
    f = fake_b.reshape(-1)
    edge_real = real_b.max(axis=-1).reshape(-1)
    edge_fake = fake_b.max(axis=-1).reshape(-1)
    out = {
        "edge_f1": float(f1_score(edge_real, edge_fake, zero_division=0)),
        "lag_f1": float(f1_score(r, f, zero_division=0)),
        "shd": float(np.mean(edge_real != edge_fake)),
        "weight_mae": float(np.mean(np.abs(real - fake))),
    }
    try:
        out["auprc"] = float(average_precision_score(r, fake.reshape(-1)))
    except ValueError:
        out["auprc"] = 0.0
    real_edge = real_b.max(axis=-1)
    fake_edge = fake_b.max(axis=-1)
    matched = (real_edge == 1) & (fake_edge == 1)
    if np.any(matched):
        real_lag = np.argmax(real * real_b, axis=-1)
        fake_lag = np.argmax(fake * fake_b, axis=-1)
        out["lag_mae"] = float(np.mean(np.abs(real_lag[matched] - fake_lag[matched])))
    else:
        out["lag_mae"] = float(real.shape[-1])
    return out


def generation_report(real: np.ndarray, fake: np.ndarray, device: torch.device, quick: bool = False) -> dict[str, float]:
    epochs = 20 if quick else 100
    pred = predictive_score(fake, real, device, epochs=epochs)
    return {
        "discriminative": discriminative_score(real, fake, device, epochs=epochs),
        "predictive_mse": pred["mse"],
        "predictive_mae": pred["mae"],
        "correlation": correlation_score(real, fake),
    }
