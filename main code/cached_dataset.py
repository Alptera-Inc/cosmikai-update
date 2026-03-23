# cached_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CacheFileInfo:
    path: Path
    target: str
    nrows: int


def read_npz_meta(path: Path) -> dict:
    """
    Reads only meta (and X shape) from a cached npz file.
    Your save_cache() stored meta as JSON under key 'meta'.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    return meta


def read_npz_xy(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads X, y, meta from npz.
    X: (K, nbins) float32
    y: (K,) float32
    """
    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)
        meta = json.loads(str(data["meta"]))
    return X, y, meta


class CachedNPZDataset(Dataset):
    """
    Dataset backed by cached .npz files created by your pipeline.
    Does NOT download anything.

    Each npz contains:
      - X: (K, nbins)
      - y: (K,)
      - meta: JSON with "target" key

    __getitem__ returns one candidate sample (x, y).
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        file_paths: Optional[List[str | Path]] = None,
        allowed_targets: Optional[List[str]] = None,
        preload: bool = False,
    ):
        self.cache_dir = Path(cache_dir)

        # Choose files
        if file_paths is None:
            paths = sorted(self.cache_dir.glob("*.npz"))
        else:
            paths = [Path(p) for p in file_paths]

        if not paths:
            raise FileNotFoundError(f"No .npz files found in {self.cache_dir.resolve()}")

        # Scan files: find target + nrows
        infos: List[CacheFileInfo] = []
        for p in paths:
            meta = read_npz_meta(p)
            target = meta.get("target", "UNKNOWN")
            # read nrows efficiently
            with np.load(p, allow_pickle=False) as data:
                nrows = int(data["X"].shape[0])

            if allowed_targets is not None and target not in set(allowed_targets):
                continue

            infos.append(CacheFileInfo(path=p, target=target, nrows=nrows))

        if not infos:
            raise ValueError("No cache files matched allowed_targets filter.")

        self.infos = infos

        # Build global index -> (file_idx, row_idx)
        self._starts: List[int] = []
        total = 0
        for info in self.infos:
            self._starts.append(total)
            total += info.nrows
        self._total = total

        # Optional preload (loads everything into RAM; faster, uses more memory)
        self.preload = preload
        self._preloaded_X: Optional[torch.Tensor] = None
        self._preloaded_y: Optional[torch.Tensor] = None

        # Lightweight last-file cache for lazy mode
        self._last_file_idx: Optional[int] = None
        self._last_X: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None

        if self.preload:
            X_all = []
            y_all = []
            for info in self.infos:
                X, y, _ = read_npz_xy(info.path)
                X_all.append(X)
                y_all.append(y)
            X_all = np.concatenate(X_all, axis=0)
            y_all = np.concatenate(y_all, axis=0)
            self._preloaded_X = torch.tensor(X_all, dtype=torch.float32)
            self._preloaded_y = torch.tensor(y_all, dtype=torch.float32)

    def __len__(self) -> int:
        return self._total

    def get_targets(self) -> List[str]:
        return sorted(set(info.target for info in self.infos))

    def get_files_by_target(self) -> Dict[str, List[Path]]:
        d: Dict[str, List[Path]] = {}
        for info in self.infos:
            d.setdefault(info.target, []).append(info.path)
        return d

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Find file index via starts (linear is fine for small lists; can be bisect if huge)
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)

        # Fast-ish locate
        # (If you end up with thousands of files, switch to bisect.)
        file_idx = 0
        for i in range(len(self._starts)):
            if i == len(self._starts) - 1 or self._starts[i + 1] > idx:
                file_idx = i
                break
        row_idx = idx - self._starts[file_idx]
        return file_idx, row_idx

    def __getitem__(self, idx: int):
        if self.preload:
            assert self._preloaded_X is not None and self._preloaded_y is not None
            return self._preloaded_X[idx], self._preloaded_y[idx]

        file_idx, row_idx = self._locate(idx)

        if self._last_file_idx != file_idx:
            X, y, _ = read_npz_xy(self.infos[file_idx].path)
            self._last_file_idx = file_idx
            self._last_X = X
            self._last_y = y

        x = torch.tensor(self._last_X[row_idx], dtype=torch.float32)
        yy = torch.tensor(self._last_y[row_idx], dtype=torch.float32)
        return x, yy