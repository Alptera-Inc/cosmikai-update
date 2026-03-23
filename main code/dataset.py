# dataset.py
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from data_ingestion import get_time_flux
from candidates import bls_topk, Candidate
from processing import fold_to_bins
from labeling import (
    sample_num_planets,
    sample_injections,
    inject_box_transits,
    label_candidates,
)

def _safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s


def cache_path(
    cache_dir: Path,
    target: str,
    mission: str,
    author: str,
    nbins: int,
    k: int,
    seed: int,
    aug_idx: int,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_safe_name(target)}_{mission}_{author}_nb{nbins}_k{k}_seed{seed}_aug{aug_idx}.npz"
    return cache_dir / fname


def build_samples_for_star(
    target: str,
    mission: str = "TESS",
    author: str = "SPOC",
    download_all: bool = False,
    nbins: int = 512,
    k: int = 15,
    dedupe_frac: float = 0.02,
    seed: int = 0,
    force_n_planets: int | None = None,
):
    """
    Part 4 wrapped into one function.

    Returns:
      X: (k, nbins) float32
      y: (k,) float32
      meta: dict (injections + candidates summary)
    """
    rng = np.random.default_rng(seed)

    time, flux = get_time_flux(target, mission, author=author, download_all=download_all)

    # Decide injection count
    n_planets = force_n_planets if force_n_planets is not None else sample_num_planets(rng)

    # Sample injections + apply them
    injections = sample_injections(time, rng, n_planets=n_planets)
    flux_inj = inject_box_transits(time, flux, injections)

    # BLS candidates on injected flux
    cands = bls_topk(time, flux_inj, k=k, dedupe_frac=dedupe_frac)

    # Fold each candidate into nbins vector
    X = np.stack(
        [fold_to_bins(time, flux_inj, c.period, c.t0, nbins=nbins) for c in cands],
        axis=0,
    ).astype(np.float32)

    # Labels (0/1)
    y = label_candidates(cands, injections, tol=0.01).astype(np.float32)

    meta = {
        "target": target,
        "mission": mission,
        "author": author,
        "seed": seed,
        "n_planets_injected": int(n_planets),
        "injections": injections,  # list of dicts
        "candidates": [asdict(c) for c in cands],  # list of dicts
    }
    return X, y, meta


def save_cache(npz_path: Path, X: np.ndarray, y: np.ndarray, meta: dict):
    meta_json = json.dumps(meta)
    np.savez_compressed(npz_path, X=X.astype(np.float32), y=y.astype(np.float32), meta=meta_json)


def load_cache(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    meta = json.loads(str(data["meta"]))
    return X, y, meta


def make_or_load_star_samples(
    target: str,
    cache_dir: Path,
    mission: str = "TESS",
    author: str = "SPOC",
    download_all: bool = False,
    nbins: int = 512,
    k: int = 15,
    dedupe_frac: float = 0.02,
    seed: int = 0,
    n_augs: int = 1,
):
    """
    For one target, generate n_augs cached files. Each aug uses seed+aug_idx.
    Returns concatenated X,y across augs.
    """
    Xs = []
    ys = []

    for aug_idx in range(n_augs):
        path = cache_path(cache_dir, target, mission, author, nbins, k, seed, aug_idx)

        if path.exists():
            X, y, _meta = load_cache(path)
        else:
            X, y, meta = build_samples_for_star(
                target=target,
                mission=mission,
                author=author,
                download_all=download_all,
                nbins=nbins,
                k=k,
                dedupe_frac=dedupe_frac,
                seed=seed + aug_idx,
            )
            save_cache(path, X, y, meta)

        Xs.append(X)
        ys.append(y)

    X_all = np.concatenate(Xs, axis=0)  # (n_augs*k, nbins)
    y_all = np.concatenate(ys, axis=0)  # (n_augs*k,)
    return X_all, y_all


class CandidateDataset(Dataset):
    """
    Eager-load dataset: builds/loads cached star samples and concatenates all candidates.
    Each item is one candidate (x, y).
    """

    def __init__(
        self,
        targets: Iterable[str],
        cache_dir: str | Path = "cache",
        mission: str = "TESS",
        author: str = "SPOC",
        download_all: bool = False,
        nbins: int = 512,
        k: int = 15,
        dedupe_frac: float = 0.02,
        seed: int = 0,
        n_augs: int = 1,
    ):
        self.cache_dir = Path(cache_dir)
        self.nbins = nbins

        X_list = []
        y_list = []

        for t in targets:
            X_star, y_star = make_or_load_star_samples(
                target=t,
                cache_dir=self.cache_dir,
                mission=mission,
                author=author,
                download_all=download_all,
                nbins=nbins,
                k=k,
                dedupe_frac=dedupe_frac,
                seed=seed,
                n_augs=n_augs,
            )
            X_list.append(X_star)
            y_list.append(y_star)

        self.X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]