from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class WindowSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scaler: StandardScaler
    train_labels: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    train_starts: Optional[np.ndarray] = None
    val_starts: Optional[np.ndarray] = None
    test_starts: Optional[np.ndarray] = None


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"x": self.windows[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def load_array(path: str | Path, label_column: str | int | None = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return _split_loaded_array(arr, label_column)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        key = "data" if "data" in data else list(data.keys())[0]
        labels = data["labels"] if "labels" in data else data["label"] if "label" in data else None
        return np.asarray(data[key], dtype=np.float32), None if labels is None else np.asarray(labels)
    df = pd.read_csv(path)
    if label_column is None:
        return df.select_dtypes(include=[np.number]).to_numpy(np.float32), None
    col = df.columns[label_column] if isinstance(label_column, int) else label_column
    labels = df[col].to_numpy(np.float32)
    values = df.drop(columns=[col]).select_dtypes(include=[np.number]).to_numpy(np.float32)
    return values, labels


def _split_loaded_array(arr: np.ndarray, label_column: str | int | None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(arr)
    if arr.ndim == 2 and isinstance(label_column, int):
        return arr[:, :label_column].astype(np.float32), arr[:, label_column].astype(np.float32)
    return arr.astype(np.float32), None


def chronological_split(
    values: np.ndarray,
    labels: Optional[np.ndarray],
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    n = values.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if values.ndim == 3:
        train = values[:n_train]
        val = values[n_train:n_train + n_val]
        test = values[n_train + n_val:]
    else:
        train = values[:n_train]
        val = values[n_train:n_train + n_val]
        test = values[n_train + n_val:]
    if labels is None:
        return train, val, test, None, None, None
    return train, val, test, labels[:n_train], labels[n_train:n_train + n_val], labels[n_train + n_val:]


def make_windows(
    values: np.ndarray,
    window_size: int,
    stride: int = 1,
    labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if values.ndim == 3:
        out = values.astype(np.float32)
        if labels is None:
            return out, None
        labels = np.asarray(labels)
        if labels.ndim == 1:
            return out, labels.astype(np.float32)
        return out, labels.max(axis=1).astype(np.float32)
    xs, ys = [], []
    for start in range(0, values.shape[0] - window_size + 1, stride):
        end = start + window_size
        xs.append(values[start:end])
        if labels is not None:
            ys.append(np.max(labels[start:end]))
    if not xs:
        empty = np.empty((0, window_size, values.shape[-1]), dtype=np.float32)
        return empty, None if labels is None else np.empty((0,), dtype=np.float32)
    windows = np.stack(xs).astype(np.float32)
    return windows, None if labels is None else np.asarray(ys, dtype=np.float32)


def make_windows_with_starts(
    values: np.ndarray,
    window_size: int,
    stride: int = 1,
    labels: Optional[np.ndarray] = None,
    start_offset: int = 0,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if values.ndim == 3:
        starts = np.arange(values.shape[0], dtype=np.int64) + start_offset
        windows, ys = make_windows(values, window_size, stride, labels)
        return windows, ys, starts
    windows, ys = make_windows(values, window_size, stride, labels)
    if len(windows) == 0:
        starts = np.empty((0,), dtype=np.int64)
    else:
        starts = np.arange(0, values.shape[0] - window_size + 1, stride, dtype=np.int64) + start_offset
    return windows, ys, starts


def standardize_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    train_labels: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    train_starts: Optional[np.ndarray] = None,
    val_starts: Optional[np.ndarray] = None,
    test_starts: Optional[np.ndarray] = None,
) -> WindowSplits:
    scaler = StandardScaler()
    fit_data = train.reshape(-1, train.shape[-1]) if train.ndim == 3 else train
    scaler.fit(fit_data)

    def tr(x: np.ndarray) -> np.ndarray:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        return scaler.transform(flat).reshape(shape).astype(np.float32)

    return WindowSplits(tr(train), tr(val), tr(test), scaler, train_labels, val_labels, test_labels, train_starts, val_starts, test_starts)


def build_window_splits(
    path: str | Path,
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    label_column: str | int | None = None,
) -> WindowSplits:
    values, labels = load_array(path, label_column)
    train, val, test, y_train, y_val, y_test = chronological_split(values, labels, train_ratio, val_ratio)
    n_train = int(values.shape[0] * train_ratio)
    n_val = int(values.shape[0] * val_ratio)
    train_w, train_y, train_s = make_windows_with_starts(train, window_size, stride, y_train, 0)
    val_w, val_y, val_s = make_windows_with_starts(val, window_size, stride, y_val, n_train)
    test_w, test_y, test_s = make_windows_with_starts(test, window_size, stride, y_test, n_train + n_val)
    return standardize_splits(train_w, val_w, test_w, train_y, val_y, test_y, train_s, val_s, test_s)


def make_loader(windows: np.ndarray, labels: Optional[np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(windows, labels), batch_size=batch_size, shuffle=shuffle, drop_last=False)
