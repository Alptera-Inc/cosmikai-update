"""
CosmiKAI Backend API Server
==============================
FastAPI REST API for the York University Satellite Capstone Team's
exoplanet transit detection ground-station interface.

Exposes a trained 1D-CNN (TransitCNN) that classifies stellar light curves
downloaded in real-time from the MAST archive via Lightkurve.  Prediction
results are cached in SQLite so repeated queries return instantly.

Endpoints
---------
    POST /api/predict   — Run transit detection on a named star
    GET  /api/history   — Paginated prediction history from the cache
    GET  /api/health    — Service health + model metadata
    GET  /api/stats     — Aggregate statistics for the telemetry dashboard

Running
-------
From the project root (cosmikai-update/):

    uvicorn backened.server:app --host 0.0.0.0 --port 8000 --reload

Environment
-----------
See ``backened/model/src/config.py`` for the full list of environment
variables.  All paths default to sensible local-development values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi import Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backened.model.src.config import (
    BLS_N_PERIODS,
    BLS_PERIOD_MAX,
    BLS_PERIOD_MIN,
    CORS_ORIGINS,
    DB_PATH,
    LOG_LEVEL,
    MODEL_PATH,
    MODEL_VERSION,
    N_BINS,
    REQUEST_TIMEOUT,
    TRAINING_SUMMARY_PATH,
    VERDICT_THRESHOLD,
)
from backened.model.src.mast_fetch import fetch_lightcurve
from backened.model.src.model import TransitCNN
from backened.model.src.preprocess import (
    bin_to_fixed_length,
    flatten_flux,
    normalize_flux,
    phase_fold,
    run_bls,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("cosmikai.server")


# ---------------------------------------------------------------------------
# Application state (loaded once at startup, shared across requests)
# ---------------------------------------------------------------------------

class _AppState:
    model: Optional[TransitCNN] = None
    model_loaded: bool = False
    training_summary: dict = {}
    start_time: float = 0.0


_state = _AppState()


# ---------------------------------------------------------------------------
# Database — initialisation and helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    star_name              TEXT    NOT NULL,
    mission                TEXT    NOT NULL,
    score                  REAL    NOT NULL,
    period_days            REAL,
    verdict                TEXT    NOT NULL,
    transit_depth_estimate REAL,
    duration_estimate      REAL,
    num_datapoints         INTEGER,
    processing_time        REAL,
    timestamp              TEXT    NOT NULL,
    UNIQUE(star_name, mission)
);
"""


async def _init_db() -> None:
    """
    Create the SQLite database file and predictions table if they don't exist.
    """
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_TABLE_SQL)
        await db.commit()
    log.info("SQLite database ready at %s", db_path.resolve())


async def _get_cached(star_name: str, mission: str) -> Optional[dict]:
    """
    Return a previously cached prediction for ``(star_name, mission)``,
    or ``None`` if no record exists.

    Parameters
    ----------
    star_name : str
        Normalised stellar identifier.
    mission : str
        Photometry mission string.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM predictions WHERE star_name = ? AND mission = ?",
            (star_name, mission),
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def _save_prediction(data: dict) -> None:
    """
    Persist a prediction result to the SQLite cache.

    Uses ``INSERT OR REPLACE`` so that ``force_rerun=True`` silently
    overwrites the previous entry for the same ``(star_name, mission)`` pair.

    Parameters
    ----------
    data : dict
        Dict produced by ``_run_pipeline_sync``.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO predictions
                (star_name, mission, score, period_days, verdict,
                 transit_depth_estimate, duration_estimate, num_datapoints,
                 processing_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["star_name"],
                data["mission"],
                data["score"],
                data.get("period_days"),
                data["verdict"],
                data.get("transit_depth_estimate"),
                data.get("duration_estimate"),
                data.get("num_datapoints"),
                data.get("processing_time"),
                data["timestamp"],
            ),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Model — loading helpers
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """
    Load TransitCNN weights from disk into ``_state.model``.

    Logs a clear error (but does not crash) if the weights file is missing
    so that the server can still start and serve cached results.
    """
    try:
        model = TransitCNN()
        weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        model.eval()
        _state.model = model
        _state.model_loaded = True
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "TransitCNN loaded — %d parameters from '%s'", n_params, MODEL_PATH
        )
    except FileNotFoundError:
        log.error(
            "Model weights not found at '%s'.  "
            "Place best_model.pt in backened/model/models/ and restart the server.",
            MODEL_PATH,
        )
        _state.model_loaded = False
    except Exception:
        log.exception("Failed to load TransitCNN weights from '%s'", MODEL_PATH)
        _state.model_loaded = False


def _load_training_summary() -> None:
    """
    Load the JSON training summary produced after model training.

    Populates ``_state.training_summary``; the server continues without it.
    """
    try:
        with open(TRAINING_SUMMARY_PATH) as fh:
            _state.training_summary = json.load(fh)
        perf = _state.training_summary.get("performance", {})
        log.info(
            "Training summary loaded — AUPRC=%.4f  stars=%s",
            perf.get("auprc", 0.0),
            _state.training_summary.get("training", {}).get("training_stars", "?"),
        )
    except FileNotFoundError:
        log.warning("training_summary.json not found at '%s'", TRAINING_SUMMARY_PATH)
        _state.training_summary = {}
    except Exception:
        log.exception("Could not parse training_summary.json")
        _state.training_summary = {}


# ---------------------------------------------------------------------------
# Prediction pipeline — fully synchronous (runs in a thread pool)
# ---------------------------------------------------------------------------

def _run_pipeline_sync(
    star_name: str,
    mission: str,
    threshold: float,
) -> dict:
    """
    Execute the complete transit detection pipeline for one star.

    This function is **synchronous** and CPU/IO bound.  It must be called
    via ``asyncio.to_thread`` so the FastAPI event loop is not blocked.

    Pipeline
    --------
    1. Download light curve from MAST (Lightkurve, may take 30–90 s)
    2. Flatten with Savitzky-Golay filter
    3. Median-normalise flux
    4. BLS period search
    5. Phase-fold to best period
    6. Median-bin to 512 points
    7. Standardise: (x − median) / std  ← CRITICAL, do not remove
    8. TransitCNN forward pass + sigmoid

    Parameters
    ----------
    star_name : str
        Stellar identifier (passed directly to Lightkurve/MAST).
    mission : str
        Photometry mission: ``"Kepler"``, ``"K2"``, or ``"TESS"``.
    threshold : float
        Verdict threshold: score ≥ threshold → TRANSIT_DETECTED.

    Returns
    -------
    dict
        Keys match the ``PredictionResponse`` Pydantic model.

    Raises
    ------
    ValueError
        For data problems (star not found, light curve too short).
    RuntimeError
        For infrastructure problems (model not loaded, BLS failure).
    """
    t_start = _time.monotonic()
    log.info("Pipeline start — star=%s  mission=%s  threshold=%.2f", star_name, mission, threshold)

    # ── 1. Download ─────────────────────────────────────────────────────────
    try:
        time_arr, flux_arr = fetch_lightcurve(star_name, mission)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"MAST download failed: {exc}") from exc

    n_points = int(len(time_arr))
    log.debug("Fetched %d datapoints for %s/%s", n_points, star_name, mission)

    if n_points < 200:
        raise ValueError(
            f"Light curve too short: only {n_points} datapoints "
            f"(minimum 200 required for reliable transit detection)."
        )

    # ── 2. Flatten ──────────────────────────────────────────────────────────
    flux_arr = flatten_flux(flux_arr, window=301, polyorder=2)

    # ── 3. Normalise ────────────────────────────────────────────────────────
    flux_arr = normalize_flux(flux_arr, method="median")

    # ── 4. BLS period search ─────────────────────────────────────────────────
    try:
        best_period = run_bls(time_arr, flux_arr, 0.6, 12.0, 5000)
    except Exception as exc:
        raise RuntimeError(f"BLS period search failed: {exc}") from exc

    log.debug("BLS — best period=%.4f d", best_period)

    # ── 5. Phase fold ────────────────────────────────────────────────────────
    phase_arr, folded_flux = phase_fold(time_arr, flux_arr, best_period)

    # ── 6. Bin ───────────────────────────────────────────────────────────────
    binned = bin_to_fixed_length(phase_arr, folded_flux, n_bins=512)

    # ── 7. Standardise — CRITICAL ────────────────────────────────────────────
    # Without this step the model outputs a near-constant ~0.5035 for all
    # inputs regardless of transit signal strength.
    med = np.median(binned)
    std = np.std(binned)
    if std > 1e-10:
        binned = (binned - med) / std
    else:
        binned = binned - med

    # ── 8. Inference ─────────────────────────────────────────────────────────
    if not _state.model_loaded or _state.model is None:
        raise RuntimeError(
            "TransitCNN is not loaded.  "
            "Ensure backened/model/models/best_model.pt exists and restart the server."
        )

    tensor = torch.from_numpy(binned).unsqueeze(0).float()  # (1, 512)
    with torch.no_grad():
        logit = _state.model(tensor)            # (1,)
        score = float(torch.sigmoid(logit).item())

    verdict = "TRANSIT_DETECTED" if score >= threshold else "NO_TRANSIT"
    processing_time = _time.monotonic() - t_start

    log.info(
        "Pipeline complete — star=%s  score=%.4f  verdict=%s  time=%.1fs",
        star_name, score, verdict, processing_time,
    )

    return {
        "star_name": star_name,
        "mission": mission,
        "score": score,
        "period_days": float(best_period),
        "verdict": verdict,
        "transit_depth_estimate": None,
        "duration_estimate": None,
        "num_datapoints": n_points,
        "processing_time": processing_time,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights and initialise the database before serving."""
    _state.start_time = _time.monotonic()
    _load_model()
    _load_training_summary()
    await _init_db()
    log.info(
        "CosmiKAI server ready — model_loaded=%s  db=%s",
        _state.model_loaded, DB_PATH,
    )
    yield
    log.info("CosmiKAI server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CosmiKAI Exoplanet Detection API",
    description=(
        "Ground-station REST API for the York University satellite capstone team. "
        "Classifies stellar light curves from MAST using a trained 1D-CNN "
        "(TransitCNN) to detect exoplanet transits in real time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for ``POST /api/predict``."""

    star_name: str = Field(
        ...,
        min_length=1,
        description="Stellar identifier (e.g. 'Kepler-10', 'TOI-700', 'KIC 11904151').",
    )
    mission: str = Field(
        ...,
        description="Photometry mission: 'Kepler', 'K2', or 'TESS'.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Verdict cutoff.  score ≥ threshold → TRANSIT_DETECTED.",
    )
    force_rerun: bool = Field(
        default=False,
        description="If True, bypass the SQLite cache and re-run the full pipeline.",
    )


class PredictionResponse(BaseModel):
    """Response body for ``POST /api/predict``."""

    star_name: str
    mission: str
    score: float = Field(description="Raw model confidence (0–1).")
    percentage: float = Field(description="score × 100, rounded to 2 dp.")
    period_days: Optional[float] = Field(description="Best BLS orbital period (days).")
    verdict: str = Field(description="'TRANSIT_DETECTED' or 'NO_TRANSIT'.")
    transit_depth_estimate: Optional[float] = Field(description="Fractional transit depth from BLS.")
    duration_estimate: Optional[float] = Field(description="Transit duration (days) from BLS.")
    num_datapoints: Optional[int] = Field(description="Photometric datapoints used.")
    cached: bool = Field(description="True if the result was served from cache.")
    processing_time_seconds: float = Field(description="Wall-clock time for the pipeline (s).")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of the prediction.")


class HistoryItem(BaseModel):
    """A single row from the predictions cache."""

    id: int
    star_name: str
    mission: str
    score: float
    percentage: float
    period_days: Optional[float]
    verdict: str
    transit_depth_estimate: Optional[float]
    duration_estimate: Optional[float]
    num_datapoints: Optional[int]
    processing_time: Optional[float]
    timestamp: str


class HistoryResponse(BaseModel):
    """Paginated response for ``GET /api/history``."""

    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    status: str
    model_loaded: bool
    model_version: str
    model_auprc: float
    total_predictions: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Response body for ``GET /api/stats``."""

    total_analyzed: int
    average_score: float
    detection_rate: float
    missions: dict[str, int]
    above_threshold: int
    below_threshold: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Run transit detection on a named star",
    tags=["inference"],
)
async def predict(body: PredictRequest) -> PredictionResponse:
    """
    Download a star's light curve from MAST and classify it with TransitCNN.

    **Cache behaviour**: if ``(star_name, mission)`` has been analysed before,
    the cached score is returned immediately and the current ``threshold``
    parameter is applied to compute the verdict.  Set ``force_rerun=True``
    to bypass the cache.

    **Processing time**: new stars typically take 30–90 s while MAST data
    downloads.  The request will time out after 180 s (configurable via
    ``COSMIKAI_REQUEST_TIMEOUT``).

    **Error codes**:
    - ``404`` — star not found in MAST or light curve too short
    - ``408`` — pipeline timed out (MAST unavailable or star has very large dataset)
    - ``503`` — model weights not loaded (check server logs)
    - ``500`` — unexpected preprocessing or inference error
    """
    star_name = body.star_name.strip()
    mission = body.mission.strip()

    if not _state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "TransitCNN model weights are not loaded.  "
                "Ensure backened/model/models/best_model.pt exists and restart the server."
            ),
        )

    log.info(
        "POST /api/predict — star=%s  mission=%s  threshold=%.2f  force=%s",
        star_name, mission, body.threshold, body.force_rerun,
    )

    # ── Cache lookup ─────────────────────────────────────────────────────────
    if not body.force_rerun:
        cached = await _get_cached(star_name, mission)
        if cached:
            log.info("Cache hit — %s / %s", star_name, mission)
            score = cached["score"]
            verdict = "TRANSIT_DETECTED" if score >= body.threshold else "NO_TRANSIT"
            return PredictionResponse(
                star_name=cached["star_name"],
                mission=cached["mission"],
                score=score,
                percentage=round(score * 100, 2),
                period_days=cached.get("period_days"),
                verdict=verdict,
                transit_depth_estimate=cached.get("transit_depth_estimate"),
                duration_estimate=cached.get("duration_estimate"),
                num_datapoints=cached.get("num_datapoints"),
                cached=True,
                processing_time_seconds=cached.get("processing_time") or 0.0,
                timestamp=cached["timestamp"],
            )

    # ── Full pipeline (in thread pool so event loop is not blocked) ──────────
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_pipeline_sync, star_name, mission, body.threshold),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            "Pipeline timed out after %.0f s for %s / %s",
            REQUEST_TIMEOUT, star_name, mission,
        )
        raise HTTPException(
            status_code=408,
            detail=(
                f"Request timed out after {int(REQUEST_TIMEOUT)} s.  "
                "MAST data download may be slow or unavailable.  Please try again."
            ),
        )
    except ValueError as exc:
        log.warning("Data error for %s / %s: %s", star_name, mission, exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        log.error("Pipeline error for %s / %s: %s", star_name, mission, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Persist to cache ──────────────────────────────────────────────────────
    await _save_prediction(result)

    return PredictionResponse(
        star_name=result["star_name"],
        mission=result["mission"],
        score=result["score"],
        percentage=round(result["score"] * 100, 2),
        period_days=result.get("period_days"),
        verdict=result["verdict"],
        transit_depth_estimate=result.get("transit_depth_estimate"),
        duration_estimate=result.get("duration_estimate"),
        num_datapoints=result.get("num_datapoints"),
        cached=False,
        processing_time_seconds=result.get("processing_time") or 0.0,
        timestamp=result["timestamp"],
    )


@app.get(
    "/api/history",
    response_model=HistoryResponse,
    summary="Paginated prediction history",
    tags=["data"],
)
async def history(
    limit: int = QueryParam(default=50, ge=1, le=200, description="Max rows to return."),
    offset: int = QueryParam(default=0, ge=0, description="Pagination offset."),
    sort: str = QueryParam(default="timestamp", description="Column to sort by."),
    order: str = QueryParam(default="desc", pattern="^(asc|desc)$", description="Sort direction."),
) -> HistoryResponse:
    """
    Return all previously analysed stars from the SQLite cache.

    Supports ``limit`` / ``offset`` pagination and sorting by any column.
    The ``total`` field in the response reflects the unfiltered row count
    and can be used to render pagination controls in the frontend.
    """
    allowed_sort = {"timestamp", "score", "star_name", "mission", "period_days"}
    if sort not in allowed_sort:
        sort = "timestamp"
    direction = "DESC" if order.lower() == "desc" else "ASC"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) FROM predictions") as cur:
            total: int = (await cur.fetchone())[0]

        async with db.execute(
            f"SELECT * FROM predictions ORDER BY {sort} {direction} LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    items = [
        HistoryItem(**{**row, "percentage": round(row["score"] * 100, 2)})
        for row in rows
    ]
    return HistoryResponse(items=items, total=total, limit=limit, offset=offset)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """
    Return service status, model metadata, and uptime.

    Useful for monitoring scripts and as a readiness probe in container
    orchestration environments.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM predictions") as cur:
            total_preds: int = (await cur.fetchone())[0]

    perf = _state.training_summary.get("performance", {})
    return HealthResponse(
        status="ok",
        model_loaded=_state.model_loaded,
        model_version=_state.training_summary.get("version", MODEL_VERSION),
        model_auprc=float(perf.get("auprc", 0.0)),
        total_predictions=total_preds,
        uptime_seconds=round(_time.monotonic() - _state.start_time, 1),
    )


@app.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="Aggregate detection statistics",
    tags=["data"],
)
async def stats(
    threshold: float = QueryParam(
        default=VERDICT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Score cutoff used to compute above/below counts.",
    )
) -> StatsResponse:
    """
    Return aggregate statistics over all cached predictions.

    Used by the **System Telemetry** tab in the React frontend.

    ``above_threshold`` is what the frontend displays as "CANDIDATE_FLAGS"
    and ``below_threshold`` as "FALSE_POSITIVES".
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) as n FROM predictions") as cur:
            total: int = (await cur.fetchone())["n"]

        if total == 0:
            return StatsResponse(
                total_analyzed=0,
                average_score=0.0,
                detection_rate=0.0,
                missions={},
                above_threshold=0,
                below_threshold=0,
            )

        async with db.execute("SELECT AVG(score) as avg FROM predictions") as cur:
            avg_score = float((await cur.fetchone())["avg"] or 0.0)

        async with db.execute(
            "SELECT COUNT(*) as n FROM predictions WHERE score >= ?", (threshold,)
        ) as cur:
            above: int = (await cur.fetchone())["n"]

        async with db.execute(
            "SELECT mission, COUNT(*) as n FROM predictions GROUP BY mission"
        ) as cur:
            missions: dict[str, int] = {
                row["mission"]: row["n"] for row in await cur.fetchall()
            }

    return StatsResponse(
        total_analyzed=total,
        average_score=round(avg_score, 4),
        detection_rate=round(above / total, 4),
        missions=missions,
        above_threshold=above,
        below_threshold=total - above,
    )
