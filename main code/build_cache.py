# build_cache.py
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np

import lightkurve as lk
from astroquery.mast import Catalogs  # pip install astroquery

# Reuse your Part 4/5 code:
from dataset import build_samples_for_star, save_cache


def safe_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s


def load_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_registry(path: Path, reg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True), encoding="utf-8")


def discover_tic_ids(
    rng: np.random.Generator,
    n_needed: int,
    tmag_min: float = 6.0,
    tmag_max: float = 11.5,
    radius_deg: float = 0.25,
    max_queries: int = 50,
) -> list[int]:
    """
    Discover TIC IDs by querying random sky points for bright stars in the TIC catalog.
    """
    found: list[int] = []
    queries = 0

    while len(found) < n_needed and queries < max_queries:
        queries += 1

        ra = float(rng.uniform(0, 360))
        dec = float(rng.uniform(-90, 90))

        # Query TIC around a random point
        tbl = Catalogs.query_region(
            f"{ra} {dec}",
            radius=radius_deg,
            catalog="TIC",
        )

        if tbl is None or len(tbl) == 0:
            continue

        # Filter by Tmag if present
        if "Tmag" in tbl.colnames:
            mask = np.isfinite(tbl["Tmag"]) & (tbl["Tmag"] >= tmag_min) & (tbl["Tmag"] <= tmag_max)
            tbl = tbl[mask]

        if len(tbl) == 0 or "ID" not in tbl.colnames:
            continue

        ids = [int(x) for x in tbl["ID"]]
        rng.shuffle(ids)

        for ticid in ids:
            found.append(ticid)
            if len(found) >= n_needed:
                break

    # de-dupe while preserving order
    seen = set()
    uniq = []
    for x in found:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def cache_path(cache_dir: Path, target: str, mission: str, author: str, nbins: int, k: int, seed: int, aug_idx: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{safe_name(target)}_{mission}_{author}_nb{nbins}_k{k}_seed{seed}_aug{aug_idx}.npz"
    return cache_dir / fname


def existing_aug_indices(cache_dir: Path, target: str, mission: str, author: str, nbins: int, k: int, seed: int) -> set[int]:
    """
    Find which aug indices already exist for this target + config.
    """
    prefix = f"{safe_name(target)}_{mission}_{author}_nb{nbins}_k{k}_seed{seed}_aug"
    used = set()
    for p in cache_dir.glob(prefix + "*.npz"):
        m = re.search(r"_aug(\d+)\.npz$", p.name)
        if m:
            used.add(int(m.group(1)))
    return used


def has_tess_spoc_data(target: str) -> tuple[bool, int]:
    """
    Quick check: does search return any SPOC light curves?
    """
    try:
        sr = lk.search_lightcurve(target, mission="TESS", author="SPOC")
        return (len(sr) > 0, len(sr))
    except Exception:
        return (False, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="targets_registry.json")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--mission", default="TESS")
    ap.add_argument("--author", default="SPOC")
    ap.add_argument("--want_targets", type=int, default=50, help="Total number of OK targets to have in registry")
    ap.add_argument("--add_augs", type=int, default=10, help="How many new augs to cache per OK target (if missing)")
    ap.add_argument("--nbins", type=int, default=512)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--dedupe_frac", type=float, default=0.02)
    ap.add_argument("--download_all", action="store_true", help="If set, download all LC files per target (slower, more SNR)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tmag_min", type=float, default=6.0)
    ap.add_argument("--tmag_max", type=float, default=11.5)
    ap.add_argument("--radius_deg", type=float, default=0.25)

    args = ap.parse_args()

    registry_path = Path(args.registry)
    cache_dir = Path(args.cache_dir)

    reg = load_registry(registry_path)

    # Count OK targets
    ok_targets = [t for t, info in reg.items() if info.get("ok") is True]
    print(f"Registry: {len(ok_targets)} ok targets, {len(reg)} total entries")

    rng = np.random.default_rng(args.seed)

    # 1) If we need more targets, discover + verify + add them
    while len(ok_targets) < args.want_targets:
        need = args.want_targets - len(ok_targets)
        print(f"Need {need} more targets. Discovering TIC IDs...")

        tic_ids = discover_tic_ids(
            rng=rng,
            n_needed=max(need * 3, 50),  # oversample to account for misses
            tmag_min=args.tmag_min,
            tmag_max=args.tmag_max,
            radius_deg=args.radius_deg,
        )

        if not tic_ids:
            print("Could not discover more TIC IDs. Try increasing --radius_deg or widening Tmag range.")
            break

        for ticid in tic_ids:
            target = f"TIC {ticid}"
            if target in reg:
                continue

            ok, nfiles = has_tess_spoc_data(target)
            reg[target] = {
                "ok": bool(ok),
                "n_files": int(nfiles),
                "checked_at": datetime.utcnow().isoformat() + "Z",
                "mission": args.mission,
                "author": args.author,
            }
            save_registry(registry_path, reg)

            if ok:
                ok_targets.append(target)
                print(f"Added OK target: {target} (n_files={nfiles})")
                if len(ok_targets) >= args.want_targets:
                    break

        print(f"Now have {len(ok_targets)} ok targets.")

    # 2) For every OK target, ensure we have enough cached augs
    print("\nCaching samples...")
    for target in ok_targets:
        used = existing_aug_indices(cache_dir, target, args.mission, args.author, args.nbins, args.k, args.seed)
        need_augs = args.add_augs

        # Choose aug indices to create
        aug_idx = 0
        created = 0
        while created < need_augs:
            if aug_idx in used:
                aug_idx += 1
                continue

            npz_path = cache_path(cache_dir, target, args.mission, args.author, args.nbins, args.k, args.seed, aug_idx)

            # Build one (X,y) chunk (k candidates) for this aug
            X, y, meta = build_samples_for_star(
                target=target,
                mission=args.mission,
                author=args.author,
                download_all=args.download_all,
                nbins=args.nbins,
                k=args.k,
                dedupe_frac=args.dedupe_frac,
                seed=args.seed + aug_idx,
            )
            save_cache(npz_path, X, y, meta)
            created += 1
            aug_idx += 1

        print(f"{target}: cached +{created} augs (now has {len(used) + created} for this config)")

    print("\nDone.")
    print(f"Registry saved to: {registry_path}")
    print(f"Cache directory:   {cache_dir}")


if __name__ == "__main__":
    main()