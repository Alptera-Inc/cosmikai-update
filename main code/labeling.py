# labeling.py
# Part 4: Transit injection + candidate labeling (for training data)

import numpy as np


# -----------------------------
# 4.1 How many planets to inject
# -----------------------------
def sample_num_planets(rng: np.random.Generator) -> int:
    """
    Returns 0, 1, 2, or 3 planets.
    Bias toward fewer planets (realistic + helps balance negatives).
    """
    # You can tune these probabilities later.
    return int(rng.choice([0, 1, 2, 3], p=[0.55, 0.30, 0.12, 0.03]))


# ---------------------------------------
# 4.2 Sample injection parameters (0–3)
# ---------------------------------------
def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_injections(
    time: np.ndarray,
    rng: np.random.Generator,
    n_planets: int,
    pmin: float = 0.8,
    pmax: float | None = None,
    min_period_separation_frac: float = 0.03,
) -> list[dict]:
    """
    Returns a list of injection dicts:
      [{"period":..., "t0":..., "duration":..., "depth":...}, ...]
    All units are in days except depth (relative flux fraction).
    """

    tmin = float(np.min(time))
    tmax = float(np.max(time))
    baseline = tmax - tmin

    if pmax is None:
        # Encourage multiple transits in the window for reliable labeling.
        pmax = max(pmin * 1.2, 0.5 * baseline)

    injections: list[dict] = []
    chosen_periods: list[float] = []

    for _ in range(n_planets):
        # Pick a period that isn't too close to an existing planet's period
        for _try in range(50):
            period = _log_uniform(rng, pmin, pmax)
            if all(abs(period - p0) / p0 > min_period_separation_frac for p0 in chosen_periods):
                chosen_periods.append(period)
                break
        else:
            # Could not find a well-separated period; stop early
            break

        # Transit center time: choose a random phase within one period
        t0 = tmin + rng.uniform(0, period)

        # Duration: rough range (in days). You can tune later.
        # 0.03–0.30 days is ~0.7h–7.2h
        duration = float(rng.uniform(0.03, 0.30))

        # Depth: choose in ppm range converted to relative flux fraction.
        # 200–20000 ppm -> 2e-4–2e-2
        depth_ppm = _log_uniform(rng, 200.0, 20000.0)
        depth = float(depth_ppm / 1e6)

        injections.append(
            {"period": float(period), "t0": float(t0), "duration": float(duration), "depth": float(depth)}
        )

    return injections


# -----------------------------
# 4.3 Inject box-shaped transits
# -----------------------------
def inject_box_transits(time: np.ndarray, flux: np.ndarray, injections: list[dict]) -> np.ndarray:
    """
    Apply multiplicative box transits: flux[in_transit] *= (1 - depth).
    Assumes flux baseline ~1 (relative flux).
    """
    out = flux.astype(np.float32).copy()

    for inj in injections:
        p = float(inj["period"])
        t0 = float(inj["t0"])
        dur = float(inj["duration"])
        depth = float(inj["depth"])

        # phase distance in days centered at 0
        phase = ((time - t0 + 0.5 * p) % p) - 0.5 * p
        in_tr = np.abs(phase) < 0.5 * dur

        out[in_tr] *= (1.0 - depth)

    return out


# ---------------------------------------------
# 4.4 Label BLS candidates by period matching
# ---------------------------------------------
def period_matches(p_cand: float, p_true: float, tol: float = 0.01) -> bool:
    """
    True if candidate period matches true period within tol (fractional),
    allowing common harmonics (half/double).
    """
    p_cand = float(p_cand)
    p_true = float(p_true)

    def close(a, b):
        return abs(a - b) / b < tol

    return close(p_cand, p_true) or close(p_cand, 0.5 * p_true) or close(p_cand, 2.0 * p_true)


def label_candidates(candidates, injections: list[dict], tol: float = 0.01) -> np.ndarray:
    """
    Best-only labeling:
    For each injected planet period, mark ONLY the highest-power matching candidate as positive.
    This prevents multiple positives from harmonics/near-duplicates for the same injected planet.
    """
    y = np.zeros(len(candidates), dtype=np.float32)

    for inj in injections:
        ptrue = float(inj["period"])

        match_idxs = []
        for i, cand in enumerate(candidates):
            if period_matches(float(cand.period), ptrue, tol=tol):
                match_idxs.append(i)

        if match_idxs:
            best_i = max(match_idxs, key=lambda i: float(candidates[i].power))
            y[best_i] = 1.0

    return y