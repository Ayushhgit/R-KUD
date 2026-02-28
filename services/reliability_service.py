"""
Merchant Reliability & Kitchen Volatility Service.

Computes:
    - MRI (Merchant Reliability Index): Composite sigmoid over marking quality features
    - KVI (Kitchen Volatility Index): Scaled rolling volatility with rush amplification

Upgrade from simple MRI = 1 / (1 + std):
    - Now uses sigmoid over weighted composite of:
        1. Inverse marking std (consistency)
        2. Mark-on-arrival rate (accuracy)
        3. Inverse delay entropy (predictability)
    - Documented real-world extension with batch delay probability

Production deployment:
    - Stateless service — reads from feature store, returns computed indices
    - Can be cached aggressively (MRI changes slowly — TTL: 5 minutes)
    - Horizontally scaled, no inter-service state
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.models import (
    KitchenVolatility,
    MerchantFeatures,
    MerchantReliability,
)
from core.optimization import compute_kvi_full, compute_mri_full

router = APIRouter(prefix="/reliability", tags=["Reliability & Volatility"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_feature_store: FeatureStore | None = None
_config: RPKOEConfig | None = None


def set_dependencies(store: FeatureStore, config: RPKOEConfig) -> None:
    """Inject dependencies at startup."""
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
    "/mri",
    response_model=MerchantReliability,
    summary="Compute Merchant Reliability Index",
    description=(
        "Computes composite MRI using sigmoid over weighted marking quality features: "
        "inverse marking std, mark-on-arrival rate, and inverse delay entropy. "
        "MRI ∈ (0, 1) where higher = more reliable merchant."
    ),
)
async def compute_mri_endpoint(
    merchant_id: str,
    features: MerchantFeatures | None = None,
) -> MerchantReliability:
    """Compute MRI for a merchant.

    Mathematical formulation:
        MRI = σ(w₁ · 1/(1 + marking_std) + w₂ · mark_on_arrival_rate + w₃ · 1/(1 + delay_entropy))

    where σ is the sigmoid function.

    Real-world extension (documented for production):
        Additional features that would improve MRI accuracy:
        - batch_delay_probability: P(merchant batches orders, causing delay)
        - peak_hour_degradation: How much reliability drops during peak
        - cancellation_initiated_rate: Merchant-side cancellation probability
    """
    store = _get_store()
    config = _get_config()

    if features is None:
        features = store.get_merchant_features(merchant_id)
        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for merchant {merchant_id}",
            )

    # Count recent FOR observations for sample_count
    sample_count = int(store.get(merchant_id, "for_sample_count", default=10))

    return compute_mri_full(features, sample_count, config)


@router.post(
    "/kvi",
    response_model=KitchenVolatility,
    summary="Compute Kitchen Volatility Index",
    description=(
        "KVI = rolling_std(pickup_delay) × external_rush_factor. "
        "Higher KVI means unpredictable kitchen output timing, "
        "amplified by external demand surges."
    ),
)
async def compute_kvi_endpoint(
    merchant_id: str,
    features: MerchantFeatures | None = None,
) -> KitchenVolatility:
    """Compute KVI for a merchant.

    Interpretation:
        KVI < 1.0: Stable kitchen — delays are predictable
        KVI 1.0-3.0: Moderate volatility — some uncertainty
        KVI > 3.0: High volatility — consider adding large safety buffer
    """
    store = _get_store()

    if features is None:
        features = store.get_merchant_features(merchant_id)
        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for merchant {merchant_id}",
            )

    return compute_kvi_full(features)


@router.get(
    "/profile/{merchant_id}",
    summary="Get full reliability profile for a merchant",
    description="Returns both MRI and KVI along with underlying features.",
)
async def get_reliability_profile(merchant_id: str) -> dict:
    """Combined reliability profile for operational dashboards."""
    store = _get_store()
    config = _get_config()

    features = store.get_merchant_features(merchant_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features found for merchant {merchant_id}",
        )

    sample_count = int(store.get(merchant_id, "for_sample_count", default=10))

    mri = compute_mri_full(features, sample_count, config)
    kvi = compute_kvi_full(features)

    return {
        "merchant_id": merchant_id,
        "mri": mri.model_dump(),
        "kvi": kvi.model_dump(),
        "features": {
            "marking_std": features.marking_std,
            "mark_on_arrival_rate": features.mark_on_arrival_rate,
            "delay_entropy": features.delay_entropy,
            "rolling_pickup_delay_std": features.rolling_pickup_delay_std,
            "external_rush_factor": features.external_rush_factor,
            "throughput_saturation": features.throughput_saturation,
        },
        "interpretation": {
            "reliability": (
                "HIGH" if mri.mri_score > 0.7 else
                "MEDIUM" if mri.mri_score > 0.5 else
                "LOW"
            ),
            "volatility": (
                "LOW" if kvi.kvi_score < 1.0 else
                "MODERATE" if kvi.kvi_score < 3.0 else
                "HIGH"
            ),
        },
    }
