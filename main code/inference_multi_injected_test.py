import numpy as np
import torch

from data_ingestion import get_time_flux
from candidates import bls_topk
from processing import fold_to_bins
from model import SmallCNN
from labeling import sample_injections, inject_box_transits

# import your detect_multi, score_candidates, mask_transit from inference_multi.py
from inference_multi import detect_multi

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

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SmallCNN(nbins).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    t, f = get_time_flux(target, mission, author=author, download_all=download_all)

    rng = np.random.default_rng(123)

    # Force exactly 2 injected planets
    injs = sample_injections(t, rng, n_planets=2, pmin=1.0, pmax=10.0)
    f_inj = inject_box_transits(t, f, injs)

    print("Injected periods:", [round(x["period"], 6) for x in injs])

    found = detect_multi(
        t, f_inj, model, device,
        nbins=nbins, k=k, dedupe_frac=dedupe_frac,
        threshold=threshold, max_planets=max_planets
    )

    print("\nDetections:")
    for i, d in enumerate(found, 1):
        print(f"{i}) prob={d['prob']:.3f} P={d['period']:.6f} dur={d['duration']:.4f}")

if __name__ == "__main__":
    main()