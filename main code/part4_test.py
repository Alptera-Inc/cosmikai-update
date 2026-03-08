import numpy as np
from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from labeling import sample_num_planets, sample_injections, inject_box_transits, label_candidates

rng = np.random.default_rng(42)

t, f = get_time_flux("Pi Mensae", "TESS", author="SPOC", download_all=False)

n_planets = sample_num_planets(rng)
injs = sample_injections(t, rng, n_planets=n_planets)
f_inj = inject_box_transits(t, f, injs)

cands = bls_topk(t, f_inj, k=15, dedupe_frac=0.02)
y = label_candidates(cands, injs, tol=0.01)

X = np.stack([fold_to_bins(t, f_inj, c.period, c.t0, nbins=512) for c in cands], axis=0)

print("Injected planets:", n_planets)
print("Injections:", injs)
print("X shape:", X.shape)
print("y positives:", int(y.sum()), "out of", len(y))
print("X finite:", np.isfinite(X).all())