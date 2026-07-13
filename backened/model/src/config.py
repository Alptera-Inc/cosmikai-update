"""
CosmiKAI — Runtime Configuration
==================================
All paths and hyperparameters are read from environment variables so that
the same codebase can run locally, in Docker, or on a satellite ground station
without code changes.  Sensible defaults are provided for local development.

Environment variables (set in shell or a .env file):
    COSMIKAI_MODEL_PATH          Path to best_model.pt
    COSMIKAI_DB_PATH             Path to SQLite cache database
    COSMIKAI_TRAINING_SUMMARY    Path to training_summary.json
    COSMIKAI_LOG_LEVEL           Python log level (DEBUG/INFO/WARNING/ERROR)
    COSMIKAI_VERDICT_THRESHOLD   Float 0–1; score >= threshold → TRANSIT_DETECTED
    COSMIKAI_REQUEST_TIMEOUT     Max seconds to wait for a pipeline run
    COSMIKAI_CORS_ORIGINS        Comma-separated list of allowed CORS origins
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory — project root (two levels above this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]  # cosmikai-update/

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.environ.get(
    "COSMIKAI_MODEL_PATH",
    str(_PROJECT_ROOT / "backened" / "model" / "models" / "best_model.pt"),
)

DB_PATH: str = os.environ.get(
    "COSMIKAI_DB_PATH",
    str(_PROJECT_ROOT / "backened" / "stars.db"),
)

TRAINING_SUMMARY_PATH: str = os.environ.get(
    "COSMIKAI_TRAINING_SUMMARY",
    str(_PROJECT_ROOT / "backened" / "model" / "models" / "training_summary.json"),
)

# ---------------------------------------------------------------------------
# Preprocessing hyperparameters — must match training exactly
# ---------------------------------------------------------------------------

#: Number of phase bins for the folded light curve fed to the CNN
N_BINS: int = int(os.environ.get("COSMIKAI_N_BINS", "512"))

#: BLS period search range (days)
BLS_PERIOD_MIN: float = float(os.environ.get("COSMIKAI_BLS_PMIN", "0.6"))
BLS_PERIOD_MAX: float = float(os.environ.get("COSMIKAI_BLS_PMAX", "12.0"))

#: Number of trial periods in the BLS grid
BLS_N_PERIODS: int = int(os.environ.get("COSMIKAI_BLS_N_PERIODS", "5000"))

#: Savitzky-Golay window length for flux flattening (must be odd; ~5 % of baseline)
FLATTEN_WINDOW: int = int(os.environ.get("COSMIKAI_FLATTEN_WINDOW", "301"))

#: Sigma threshold for outlier rejection inside the flattening step
FLATTEN_SIGMA: int = int(os.environ.get("COSMIKAI_FLATTEN_SIGMA", "2"))

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------

#: Default classification threshold applied to the raw sigmoid score
VERDICT_THRESHOLD: float = float(os.environ.get("COSMIKAI_VERDICT_THRESHOLD", "0.5"))

#: Seconds before a pipeline run is cancelled (MAST can be slow)
REQUEST_TIMEOUT: float = float(os.environ.get("COSMIKAI_REQUEST_TIMEOUT", "380.0"))

# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("COSMIKAI_LOG_LEVEL", "INFO").upper()

_raw_origins = os.environ.get(
    "COSMIKAI_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:8080",
)
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

#: Human-readable version shown in /api/health
MODEL_VERSION: str = os.environ.get("COSMIKAI_MODEL_VERSION", "1.0.0")
