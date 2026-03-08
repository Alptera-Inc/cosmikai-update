# inference_single.py
from __future__ import annotations

import numpy as np
import torch

from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from model import SmallCNN


@torch.no_grad()
def score_candidates(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    """
    X: (K, nbins) float32
    returns probs: (K,) float32
    """
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xt)
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return probs


def main():
    # -------- settings --------
    checkpoint_path = r"checkpoints\best_smallcnn.pt"
    target = "Pi Mensae"
    mission = "TESS"
    author = "SPOC"
    download_all = False

    nbins = 512
    k = 15
    dedupe_frac = 0.02

    top_n_print = 10  # how many candidates to print
    # --------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Your SmallCNN expects positional nbins based on earlier Fix A
    model = SmallCNN(nbins).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded checkpoint from epoch:", ckpt.get("epoch", "?"), "val_recall:", ckpt.get("val_recall", "?"))

    # Get data
    t, f = get_time_flux(target, mission, author=author, download_all=download_all)

    # BLS candidates
    cands = bls_topk(t, f, k=k, dedupe_frac=dedupe_frac)

    # Build X from folded candidates
    X = np.stack([fold_to_bins(t, f, c.period, c.t0, nbins=nbins) for c in cands], axis=0).astype(np.float32)
    assert np.isfinite(X).all(), "X contains NaN/Inf; check fold_to_bins."

    # Score with model
    probs = score_candidates(model, X, device)

    # Sort by model probability (descending)
    order = np.argsort(probs)[::-1]

    print("\nTop candidates by MODEL probability:")
    for rank, idx in enumerate(order[:top_n_print], 1):
        c = cands[idx]
        print(
            f"{rank:2d}) prob={probs[idx]:.3f} | "
            f"P={c.period:.6f} d | dur={c.duration:.4f} d | depth={c.depth:.3e} | bls_power={c.power:.3e}"
        )

    # Also show best BLS candidate (for comparison)
    best_bls = max(range(len(cands)), key=lambda i: cands[i].power)
    print("\nBest candidate by BLS power:")
    c = cands[best_bls]
    print(
        f"prob={probs[best_bls]:.3f} | "
        f"P={c.period:.6f} d | dur={c.duration:.4f} d | depth={c.depth:.3e} | bls_power={c.power:.3e}"
    )


if __name__ == "__main__":
    main()