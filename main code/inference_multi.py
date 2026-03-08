# inference_multi.py
import numpy as np
import torch

from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from model import SmallCNN


@torch.no_grad()
def score_candidates(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xt)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def mask_transit(time, flux, period, t0, duration, rng=None):
    """
    Replace in-transit points with out-of-transit median + noise
    to avoid creating a flat artificial feature.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    out = flux.copy().astype(np.float32)
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_tr = np.abs(phase) < 0.5 * duration

    oot = out[~in_tr]
    med = np.nanmedian(oot) if oot.size > 0 else np.nanmedian(out)
    std = np.nanstd(oot) if oot.size > 5 else (np.nanstd(out) + 1e-6)

    out[in_tr] = med + rng.normal(0.0, std, size=int(in_tr.sum())).astype(np.float32)
    return out


def is_related_period(p, fp, frac=0.03):
    ratios = [1.0, 0.5, 2.0, 1/3, 3.0, 2/3, 3/2]
    for r in ratios:
        target = fp * r
        if target > 0 and abs(p - target) / target < frac:
            return True
    return False


def is_duplicate_or_harmonic(p, found_periods, frac=0.03):
    return any(is_related_period(p, fp, frac=frac) for fp in found_periods)


def detect_multi(
    time, flux, model, device,
    nbins=512, k=30, dedupe_frac=0.02,
    threshold=0.35, max_planets=5,
    related_frac=0.04,
):
    """
    Multi-planet detector:
    - BLS proposes candidates
    - CNN gates candidates by probability
    - Choose strongest-by-BLS among those passing gates & not harmonic of previous
    - Mask detected transits and repeat

    Power gates:
    - Strict on first detection
    - Looser on later detections (helps recover weaker second planets)
    """
    residual = flux.astype(np.float32).copy()
    found = []
    found_periods = []
    rng = np.random.default_rng(2024)

    initial_best_power = None

    for _ in range(max_planets):
        cands = bls_topk(time, residual, k=k, dedupe_frac=dedupe_frac)

        X = np.stack(
            [fold_to_bins(time, residual, c.period, c.t0, nbins=nbins) for c in cands],
            axis=0
        ).astype(np.float32)

        probs = score_candidates(model, X, device)

        best_power_iter = max(float(c.power) for c in cands)
        if initial_best_power is None:
            initial_best_power = best_power_iter

        # 1) CNN probability gate (optionally looser after first planet)
        thr = threshold if len(found) == 0 else max(0.25, threshold - 0.10)
        idxs = [i for i in range(len(cands)) if float(probs[i]) >= thr]
        if not idxs:
            break

        # 2) Power gates: strict for first detection, looser for later detections
        if len(found) == 0:
            pfi = 0.60  # >= 60% of this-iteration best power
            pfo = 0.40  # >= 40% of initial best power
        else:
            pfi = 0.45  # looser after first planet
            pfo = 0.25

        idxs = [
            i for i in idxs
            if float(cands[i].power) >= pfi * best_power_iter
            and float(cands[i].power) >= pfo * initial_best_power
        ]
        if not idxs:
            break

        # Prefer strongest periodic evidence among those that pass the gates
        idxs.sort(key=lambda i: float(cands[i].power), reverse=True)

        chosen = None
        for i in idxs:
            p = float(cands[i].period)
            if is_duplicate_or_harmonic(p, found_periods, frac=related_frac):
                continue
            chosen = i
            break

        if chosen is None:
            break

        c = cands[chosen]
        found.append({
            "prob": float(probs[chosen]),
            "period": float(c.period),
            "t0": float(c.t0),
            "duration": float(c.duration),
            "depth": float(c.depth),
            "bls_power": float(c.power),
        })
        found_periods.append(float(c.period))

        residual = mask_transit(time, residual, c.period, c.t0, c.duration, rng=rng)

    return found


def main():
    checkpoint_path = r"checkpoints\best_smallcnn.pt"
    target = "Pi Mensae"
    mission = "TESS"
    author = "SPOC"
    download_all = False

    nbins = 512
    k = 30
    dedupe_frac = 0.02
    threshold = 0.35
    max_planets = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SmallCNN(nbins).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    t, f = get_time_flux(target, mission, author=author, download_all=download_all)

    found = detect_multi(
        t, f, model, device,
        nbins=nbins, k=k, dedupe_frac=dedupe_frac,
        threshold=threshold, max_planets=max_planets
    )

    print("\nDetections:")
    if not found:
        print("  None above threshold.")
    else:
        for i, d in enumerate(found, 1):
            print(
                f"{i}) prob={d['prob']:.3f} | P={d['period']:.6f} d | dur={d['duration']:.4f} d | "
                f"depth={d['depth']:.3e} | bls_power={d['bls_power']:.3e}"
            )


if __name__ == "__main__":
    main()