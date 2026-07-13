"""
CosmiKAI — MAST Data Fetcher
==============================
Downloads stellar light curves from MAST via Lightkurve.
Falls back to Target Pixel Files if no preprocessed light curve exists.
"""

import lightkurve as lk
import numpy as np


def fetch_lightcurve(star_name, mission="Kepler"):
    """
    Fetch light curve from MAST via Lightkurve.
    Falls back to Target Pixel Files if no light curve product exists.
    Returns (time, flux) as numpy arrays.
    """
    sr = lk.search_lightcurve(star_name, mission=mission)
    if len(sr) > 0:
        lc = sr.download_all().stitch().remove_nans()
        return lc.time.value.astype(np.float32), lc.flux.value.astype(np.float32)

    sr_tpf = lk.search_targetpixelfile(star_name, mission=mission)
    if len(sr_tpf) > 0:
        tpf = sr_tpf[0].download()
        lc = tpf.to_lightcurve(aperture_mask="pipeline")
        lc = lc.remove_nans()
        if len(lc.flux) < 500:
            raise ValueError(f"TPF light curve too short for {star_name} ({len(lc.flux)} pts)")
        return lc.time.value.astype(np.float32), lc.flux.value.astype(np.float32)

    raise ValueError(f"No light curves or pixel files found for {star_name}")