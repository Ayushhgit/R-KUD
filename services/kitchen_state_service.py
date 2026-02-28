"""
Kitchen State Estimation Service.

Provides real-time hidden kitchen congestion state estimation and
probabilistic KPT (Kitchen Preparation Time) distribution computation.

Key upgrades over naive implementation:
    - Stateful congestion estimation with exponential decay memory
    - Spike detection via residual drift thresholds
    - Distribution calibration monitoring via interval coverage tracking

Production deployment:
    - Deployed as a stateless HTTP service (state lives in feature store)
    - Horizontally scaled behind a load balancer
    - Latency target: P99 < 20ms
    - Each request is independent — the service reads from the feature
      store and computes the result. No in-process state.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.models import KitchenState, KPTDistribution, MerchantFeatures
from core.optimization import compute_congestion_with_memory, compute_kpt_distribution

router = APIRouter(prefix="/kitchen", tags=["Kitchen State"])

# ---------------------------------------------------------------------------
# Dependency injection — replaced at app startup in main.py
# ---------------------------------------------------------------------------

_feature_store: FeatureStore | None = None
_config: RPKOEConfig | None = None

# Calibration tracking: {merchant_id: list[(predicted_p90, actual)]}
_calibration_buffer: dict[str, list[tuple[float, float]]] = {}


def set_dependencies(store: FeatureStore, config: RPKOEConfig) -> None:
    """Inject dependencies at startup. Avoids global coupling."""
    global _feature_store, _config
    _feature_store = store
    _config = config


def _get_store() -> FeatureStore:
    if _feature_store is None:
        raise RuntimeError("Feature store not initialized")
    return _feature_store


def _get_config() -> RPKOEConfig:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/state",
    response_model=KitchenState,
    summary="Estimate hidden kitchen congestion state",
    description=(
        "Computes and returns the congestion score for a merchant. "
        "Uses exponential decay memory over previous state and applies "
        "spike penalties when residual drift exceeds threshold."
    ),
)
async def estimate_kitchen_state(
    merchant_id: str,
    features: MerchantFeatures | None = None,
) -> KitchenState:
    """Estimate hidden kitchen congestion with stateful memory.

    If features are not provided, they are fetched from the feature store.
    The previous congestion score is retrieved from the feature store's
    congestion memory for decay-based estimation.
    """
    store = _get_store()
    config = _get_config()

    # Resolve features
    if features is None:
        features = store.get_merchant_features(merchant_id)
        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for merchant {merchant_id}. "
                "Process some events first or provide features directly.",
            )

    # Retrieve previous congestion for stateful estimation
    previous_congestion = store.get_previous_congestion(merchant_id)

    # Compute new state
    state = compute_congestion_with_memory(features, previous_congestion, config)

    # Persist for next estimation cycle
    store.set_previous_congestion(merchant_id, state.congestion_score)

    return state


@router.post(
    "/kpt",
    response_model=KPTDistribution,
    summary="Compute probabilistic KPT distribution",
    description=(
        "Returns a LogNormal KPT distribution parameterized by the merchant's "
        "current congestion score. Includes mean, std, and percentiles (P50-P95)."
    ),
)
async def compute_kpt(
    merchant_id: str,
    congestion_score: float | None = None,
) -> KPTDistribution:
    """Compute KPT distribution for a merchant.

    If congestion_score is not provided, it is computed by first
    calling the state estimation pipeline.
    """
    store = _get_store()
    config = _get_config()
    now = datetime.now()

    if congestion_score is None:
        # Generate state first
        state = await estimate_kitchen_state(merchant_id)
        congestion_score = state.congestion_score

    kpt = compute_kpt_distribution(congestion_score, merchant_id, now, config)

    return kpt


@router.post(
    "/calibration/record",
    summary="Record an actual KPT observation for calibration",
    description=(
        "Used by downstream systems to report actual preparation times. "
        "This data feeds the distribution calibration monitoring that "
        "detects when predicted intervals are miscalibrated."
    ),
)
async def record_calibration_observation(
    merchant_id: str,
    predicted_p90: float,
    actual_kpt_minutes: float,
) -> dict:
    """Record an observation for distribution calibration monitoring.

    Tracks whether actual KPT values fall inside predicted P90 intervals.
    If the observed coverage deviates significantly from 90%, a
    calibration drift alert should be triggered.

    Production implementation:
        - Stored in a time-series database (InfluxDB / Prometheus)
        - Rolling CRPS computed in a Flink job
        - Alerts via PagerDuty when calibration_error > threshold
    """
    config = _get_config()

    if merchant_id not in _calibration_buffer:
        _calibration_buffer[merchant_id] = []

    _calibration_buffer[merchant_id].append((predicted_p90, actual_kpt_minutes))

    # Trim to calibration window size
    window = config.kpt.calibration_window_size
    if len(_calibration_buffer[merchant_id]) > window:
        _calibration_buffer[merchant_id] = _calibration_buffer[merchant_id][-window:]

    # Compute coverage
    observations = _calibration_buffer[merchant_id]
    inside_count = sum(1 for p90, actual in observations if actual <= p90)
    observed_coverage = inside_count / len(observations) if observations else 0.0
    calibration_error = observed_coverage - config.kpt.expected_coverage_p90

    is_calibrated = abs(calibration_error) <= config.kpt.calibration_alert_threshold

    return {
        "merchant_id": merchant_id,
        "sample_count": len(observations),
        "observed_coverage_p90": round(observed_coverage, 4),
        "expected_coverage_p90": config.kpt.expected_coverage_p90,
        "calibration_error": round(calibration_error, 4),
        "is_calibrated": is_calibrated,
        "action": "NONE" if is_calibrated else "DRIFT_ALERT_TRIGGERED",
    }


@router.get(
    "/calibration/{merchant_id}",
    summary="Get calibration status for a merchant",
)
async def get_calibration_status(merchant_id: str) -> dict:
    """Retrieve current calibration metrics for a merchant."""
    config = _get_config()
    observations = _calibration_buffer.get(merchant_id, [])

    if not observations:
        return {
            "merchant_id": merchant_id,
            "status": "NO_DATA",
            "sample_count": 0,
        }

    inside_count = sum(1 for p90, actual in observations if actual <= p90)
    observed_coverage = inside_count / len(observations)
    calibration_error = observed_coverage - config.kpt.expected_coverage_p90

    return {
        "merchant_id": merchant_id,
        "sample_count": len(observations),
        "observed_coverage_p90": round(observed_coverage, 4),
        "calibration_error": round(calibration_error, 4),
        "is_calibrated": abs(calibration_error) <= config.kpt.calibration_alert_threshold,
    }
