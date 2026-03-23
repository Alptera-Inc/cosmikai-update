# tune_threshold.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch

from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from model import SmallCNN
from labeling import sample_injections, inject_box_transits


def load_ok_targets_from_registry(path="targets_registry.json") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    return [t for t, info in reg.items() if info.get("ok") is True]


def period_matches_with_harmonics(p_cand: float, p_true: float, tol=0.01) -> bool:
    ratios = [1.0, 0.5, 2.0, 1 / 3, 3.0, 2 / 3, 3 / 2]
    for r in ratios:
        pt = p_true * r
        if pt > 0 and abs(p_cand - pt) / pt < tol:
            return True
    return False


@torch.no_grad()
def score_candidates(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xt)
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def pick_first_planet_candidate(
    cands,
    probs: np.ndarray,
    threshold: float,
    power_gate_frac: float = 0.60,
) -> int | None:
    """
    Returns index of chosen first-planet candidate or None.
    Strategy: CNN gate + strong BLS power gate, then choose highest BLS power.
    """
    best_power = max(float(c.power) for c in cands)
    idxs = [
        i for i in range(len(cands))
        if float(probs[i]) >= threshold and float(cands[i].power) >= power_gate_frac * best_power
    ]
    if not idxs:
        return None
    idxs.sort(key=lambda i: float(cands[i].power), reverse=True)
    return idxs[0]


def evaluate_thresholds(
    targets: list[str],
    checkpoint_path: str,
    mission="TESS",
    author="SPOC",
    download_all=False,
    nbins=512,
    bls_k=200,
    dedupe_frac=0.02,
    tol=0.01,
    trials=60,
    thresholds=np.arange(0.15, 0.86, 0.05),
    seed=123,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SmallCNN(nbins).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    rng = np.random.default_rng(seed)

    # in-memory cache so we don't re-download the same target every trial
    tf_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Trial plan: include some no-planet trials to measure false positives
    # You can tweak these probabilities
    n_planet_choices = [0, 1, 2]
    n_planet_probs = [0.45, 0.45, 0.10]

    # Stats per threshold
    results = {float(th): {"tp": 0, "fp": 0, "fn": 0, "trials_0": 0, "trials_pos": 0} for th in thresholds}

    for t_i in range(trials):
        target = str(rng.choice(targets))

        if target not in tf_cache:
            t, f = get_time_flux(target, mission, author=author, download_all=download_all)
            tf_cache[target] = (t, f)
        else:
            t, f = tf_cache[target]

        n_planets = int(rng.choice(n_planet_choices, p=n_planet_probs))

        # Inject 1–2 planets into real data (for test)
        injs = sample_injections(t, rng, n_planets=n_planets, pmin=1.0, pmax=10.0)
        f_inj = inject_box_transits(t, f, injs)

        # BLS candidates
        cands = bls_topk(t, f_inj, k=bls_k, dedupe_frac=dedupe_frac)
        X = np.stack([fold_to_bins(t, f_inj, c.period, c.t0, nbins=nbins) for c in cands], axis=0).astype(np.float32)
        probs = score_candidates(model, X, device)

        true_periods = [float(inj["period"]) for inj in injs]

        for th in thresholds:
            th = float(th)
            chosen = pick_first_planet_candidate(cands, probs, threshold=th, power_gate_frac=0.60)

            if n_planets == 0:
                results[th]["trials_0"] += 1
                if chosen is None:
                    # correct negative (we don't track TN explicitly)
                    pass
                else:
                    results[th]["fp"] += 1
            else:
                results[th]["trials_pos"] += 1
                if chosen is None:
                    results[th]["fn"] += 1
                else:
                    p_found = float(cands[chosen].period)
                    ok = any(period_matches_with_harmonics(p_found, pt, tol=tol) for pt in true_periods)
                    if ok:
                        results[th]["tp"] += 1
                    else:
                        # It predicted a planet but not matching injected one => treat as FP
                        results[th]["fp"] += 1

    # Summarize
    summary = []
    for th in thresholds:
        th = float(th)
        tp = results[th]["tp"]
        fp = results[th]["fp"]
        fn = results[th]["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # False positive rate on zero-planet trials
        fpr = fp / max(results[th]["trials_0"], 1)

        summary.append((th, precision, recall, f1, fpr))

    # Choose best: prioritize F1, break ties by lower FPR
    summary.sort(key=lambda x: (x[3], -x[4]), reverse=True)

    print("\nThreshold tuning results (sorted by F1 desc):")
    print("thr   precision  recall   f1      false_pos_rate_on_0planet")
    for th, p, r, f1, fpr in summary[:12]:
        print(f"{th:0.2f}   {p:0.3f}      {r:0.3f}   {f1:0.3f}   {fpr:0.3f}")

    best = summary[0]
    print("\nRecommended threshold (first planet):", best[0])
    return best[0]


def main():
    targets = load_ok_targets_from_registry("targets_registry.json")
    if len(targets) < 5:
        raise ValueError("Need at least ~5 ok targets in registry to tune thresholds.")

    best_thr = evaluate_thresholds(
        targets=targets,
        checkpoint_path=r"checkpoints\best_smallcnn.pt",
        trials=60,                 # increase later (e.g., 200+) for more stable result
        bls_k=200,                 # larger helps candidate coverage
        thresholds=np.arange(0.15, 0.86, 0.05),
        download_all=False,        # keep fast; can set True for higher SNR
    )

    # Optional: write it out
    with open("threshold_recommendation.json", "w", encoding="utf-8") as f:
        json.dump({"recommended_threshold_first": best_thr}, f, indent=2)
    print("Wrote threshold_recommendation.json")


if __name__ == "__main__":
    main()