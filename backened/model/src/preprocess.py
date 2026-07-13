"""
CosmiKAI — Preprocessing Pipeline
===================================
Must match the exact functions used during training.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter


def flatten_flux(flux, window=301, polyorder=2):
    """Remove stellar/instrumental trends with Savitzky-Golay filter."""
    if window % 2 == 0:
        window += 1
    trend = savgol_filter(flux, window, polyorder)
    return flux / trend


def normalize_flux(flux, method="median"):
    """Scale flux so baseline is approximately 1."""
    if method == "median":
        return flux / np.median(flux)
    elif method == "zscore":
        return (flux - np.mean(flux)) / np.std(flux)
    else:
        raise ValueError("Unknown normalization method")


def run_bls(time, flux, period_min=0.6, period_max=12.0, n_periods=5000):
    """Box Least Squares period search. Returns best period in days."""
    periods = np.linspace(period_min, period_max, n_periods)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, 0.05)
    best_period = result.period[np.argmax(result.power)]
    return best_period


def phase_fold(time, flux, period):
    """Fold light curve on period and sort by phase."""
    phase = (time % period) / period
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def bin_to_fixed_length(phase, flux, n_bins=512):
    """Bin phase-folded light curve into fixed number of bins."""
    phase = np.asarray(phase)
    flux = np.asarray(flux)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(phase, bins) - 1
    mask = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[mask], weights=flux[mask], minlength=n_bins)
    counts = np.bincount(idx[mask], minlength=n_bins)
    counts = np.maximum(counts, 1)
    binned = sums / counts
    return binned.astype(np.float32)