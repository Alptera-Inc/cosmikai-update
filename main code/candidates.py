# This files makes the candidates class so we can store the candidates easily

# last updated: 23-feb-2026
# updated by: Biswaprakash Nayak
# changes made: creating the file and making the class

# heard its better to use dataclasses than normal classes, tho idk if it really is better
import numpy as np
from dataclasses import dataclass
from astropy.timeseries import BoxLeastSquares

# makes each candidate a dataclass for better organization and readability
@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float
    power: float

def prep_flux_for_bls(flux: np.ndarray) -> np.ndarray:
    # Your flux is ~1 baseline; BLS behaves better on zero-centered data
    return (flux - np.nanmedian(flux)).astype(np.float32)

def bls_topk(
    time: np.ndarray,
    flux: np.ndarray,
    k: int = 15,
    pmin: float = 0.5,
    pmax: float | None = None,
    n_periods: int = 20000,
    durations: np.ndarray | None = None,
    dedupe_frac: float = 0.02,
) -> list[Candidate]:

    time = np.asarray(time, dtype=np.float64)
    y = prep_flux_for_bls(np.asarray(flux, dtype=np.float32))

    baseline = float(time.max() - time.min())
    if pmax is None:
        pmax = max(1.0, 0.8 * baseline)

    if durations is None:
        durations = np.linspace(0.03, 0.3, 20).astype(np.float64)

    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), n_periods)).astype(np.float64)

    bls = BoxLeastSquares(time, y)
    power = bls.power(periods, durations)

    order = np.argsort(power.power)[::-1]

    picked: list[Candidate] = []

    def is_duplicate(p: float) -> bool:
        return any(abs(p - c.period) / c.period < dedupe_frac for c in picked)

    for idx in order:
        p = float(power.period[idx])
        if is_duplicate(p):
            continue

        picked.append(Candidate(
            period=p,
            t0=float(power.transit_time[idx]),
            duration=float(power.duration[idx]),
            depth=float(power.depth[idx]),
            power=float(power.power[idx]),
        ))

        if len(picked) >= k:
            break

    return picked