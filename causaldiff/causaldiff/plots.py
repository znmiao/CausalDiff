from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def window_embedding_plots(real: np.ndarray, fake: np.ndarray, out_dir: str | Path, prefix: str = "distribution") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = min(len(real), len(fake))
    x = np.concatenate([real[:n].reshape(n, -1), fake[:n].reshape(n, -1)], axis=0)
    y = np.array([0] * n + [1] * n)
    _scatter(PCA(n_components=2).fit_transform(x), y, out / f"{prefix}_pca.png", "PCA")
    perplexity = max(2, min(30, n // 2))
    _scatter(TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(x), y, out / f"{prefix}_tsne.png", "t-SNE")


def causal_heatmap(causal: np.ndarray, out_path: str | Path) -> None:
    mat = causal.mean(axis=0).max(axis=-1) if causal.ndim == 4 else causal
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _scatter(z: np.ndarray, y: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.scatter(z[y == 0, 0], z[y == 0, 1], s=8, alpha=0.7, label="Real")
    plt.scatter(z[y == 1, 0], z[y == 1, 1], s=8, alpha=0.7, label="Generated")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
