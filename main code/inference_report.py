# inference_report.py
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares

from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from model import SmallCNN


def safe_dir_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s


@torch.no_grad()
def score_candidates(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xt)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def bls_periodogram(time: np.ndarray, flux: np.ndarray, pmin=0.5, pmax=None, n_periods=8000):
    """
    For plotting only (full power curve).
    """
    time = np.asarray(time, dtype=np.float64)
    y = (np.asarray(flux, dtype=np.float32) - np.nanmedian(flux)).astype(np.float32)

    baseline = float(time.max() - time.min())
    if pmax is None:
        pmax = max(1.0, 0.8 * baseline)

    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), n_periods)).astype(np.float64)
    durations = np.linspace(0.03, 0.3, 20).astype(np.float64)

    bls = BoxLeastSquares(time, y)
    power = bls.power(periods, durations)
    return power.period.astype(np.float64), power.power.astype(np.float64)


def write_candidates_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rank", "prob", "period", "duration", "t0", "depth", "bls_power"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def plot_folded(path: Path, pf: np.ndarray, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(pf)
    plt.title(title)
    plt.xlabel("phase bin")
    plt.ylabel("standardized flux")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_periodogram(path: Path, periods: np.ndarray, power: np.ndarray, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(periods, power)
    plt.xscale("log")
    plt.title(title)
    plt.xlabel("Period (days)")
    plt.ylabel("BLS power")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_report(
    target: str,
    checkpoint_path: str = r"checkpoints\best_smallcnn.pt",
    out_dir: str | Path = "reports",
    mission="TESS",
    author="SPOC",
    download_all=False,
    nbins=512,
    k=30,
    dedupe_frac=0.02,
    top_n=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SmallCNN(nbins).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    time, flux = get_time_flux(target, mission, author=author, download_all=download_all)

    cands = bls_topk(time, flux, k=k, dedupe_frac=dedupe_frac)
    X = np.stack([fold_to_bins(time, flux, c.period, c.t0, nbins=nbins) for c in cands], axis=0).astype(np.float32)
    probs = score_candidates(model, X, device)

    # Sort by prob desc
    order = np.argsort(probs)[::-1]
    rows = []
    for rnk, idx in enumerate(order[:top_n], 1):
        c = cands[idx]
        rows.append({
            "rank": rnk,
            "prob": float(probs[idx]),
            "period": float(c.period),
            "duration": float(c.duration),
            "t0": float(c.t0),
            "depth": float(c.depth),
            "bls_power": float(c.power),
        })

    base = Path(out_dir) / safe_dir_name(target)
    base.mkdir(parents=True, exist_ok=True)

    # Save CSV + JSON
    write_candidates_csv(base / "candidates.csv", rows)
    with open(base / "report.json", "w", encoding="utf-8") as f:
        json.dump({
            "target": target,
            "mission": mission,
            "author": author,
            "download_all": download_all,
            "nbins": nbins,
            "k": k,
            "top_n": top_n,
            "candidates": rows,
        }, f, indent=2)

    # Plots: folded top candidate
    best = rows[0]
    pf = fold_to_bins(time, flux, best["period"], best["t0"], nbins=nbins)
    plot_folded(base / "folded_top.png", pf, title=f"{target} | top prob={best['prob']:.3f} | P={best['period']:.4f}d")

    # Plot: BLS periodogram
    per, powr = bls_periodogram(time, flux)
    plot_periodogram(base / "bls_periodogram.png", per, powr, title=f"{target} | BLS periodogram")

    print(f"Saved report -> {base.resolve()}")


def main():
    # Quick single target run:
    run_report("Pi Mensae")


if __name__ == "__main__":
    main()