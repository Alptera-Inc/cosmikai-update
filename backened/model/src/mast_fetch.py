"""
CosmiKAI — deterministic MAST light-curve selection.

Only official pipeline products at one cadence are combined for each mission.
This avoids stitching overlapping community products and mixed cadences, which
can multiply the sample count and make inference unnecessarily slow.
"""

from __future__ import annotations

import logging

import lightkurve as lk
import numpy as np


log = logging.getLogger("cosmikai.mast_fetch")

# Keep inference inputs deterministic and mission-specific.  Kepler and K2
# models use the official 30-minute products.  For TESS, use official SPOC
# 1/2-minute target products and avoid mixing them with FFI/community curves.
_PRODUCT_FILTERS: dict[str, dict[str, str | int]] = {
    "Kepler": {"author": "Kepler", "exptime": 1800},
    "K2": {"author": "K2", "exptime": 1800},
    "TESS": {"author": "SPOC", "exptime": "short"},
}


def _canonical_mission(mission: str) -> str:
    value = mission.strip().upper()
    names = {"KEPLER": "Kepler", "K2": "K2", "TESS": "TESS"}
    try:
        return names[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported mission '{mission}'. Expected Kepler, K2, or TESS."
        ) from exc


def _clean_time_flux(light_curve) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, time-sorted arrays with duplicate timestamps removed."""
    time = np.asarray(light_curve.time.value, dtype=np.float64)
    flux = np.asarray(light_curve.flux.value, dtype=np.float64)

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    order = np.argsort(time, kind="stable")
    time = time[order]
    flux = flux[order]

    # Products from adjacent observing windows can occasionally share a
    # boundary cadence. Keep one sample for each timestamp before BLS.
    time, unique_indices = np.unique(time, return_index=True)
    flux = flux[unique_indices]

    return time.astype(np.float32), flux.astype(np.float32)


def fetch_lightcurve(star_name: str, mission: str = "Kepler") -> tuple[np.ndarray, np.ndarray]:
    """Download one consistent set of official light curves from MAST."""
    mission = _canonical_mission(mission)
    filters = _PRODUCT_FILTERS[mission]

    search = lk.search_lightcurve(star_name, mission=mission, **filters)
    if len(search) > 0:
        log.info(
            "MAST selection — target=%s mission=%s author=%s exptime=%s products=%d",
            star_name,
            mission,
            filters["author"],
            filters["exptime"],
            len(search),
        )
        collection = search.download_all()
        if collection is None or len(collection) == 0:
            raise RuntimeError(f"MAST returned no downloadable light curves for {star_name}")
        light_curve = collection.stitch().remove_nans()
        time, flux = _clean_time_flux(light_curve)
        if len(time) < 200:
            raise ValueError(
                f"Official {mission} light curve too short for {star_name} "
                f"({len(time)} points)"
            )
        return time, flux

    # Fall back to a matching official target-pixel product, but do not fall
    # back to unfiltered community products or mixed cadences.
    pixel_search = lk.search_targetpixelfile(star_name, mission=mission, **filters)
    if len(pixel_search) > 0:
        target_pixels = pixel_search[0].download()
        if target_pixels is None:
            raise RuntimeError(f"MAST target-pixel download failed for {star_name}")
        light_curve = target_pixels.to_lightcurve(aperture_mask="pipeline").remove_nans()
        time, flux = _clean_time_flux(light_curve)
        if len(time) < 500:
            raise ValueError(
                f"Target-pixel light curve too short for {star_name} ({len(time)} points)"
            )
        return time, flux

    raise ValueError(
        f"No official {mission} {filters['exptime']}-cadence products found for {star_name}"
    )
