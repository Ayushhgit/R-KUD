"""
Uncertainty-Aware Dispatch Control Engine.

The crown jewel of RPKOE. Orchestrates kitchen state estimation,
reliability scoring, and KPT distribution to make adaptive rider
dispatch decisions with full explainability.

Key features:
    - Supply-aware safety buffer adjustment
    - Risk level classification (LOW / MODERATE / HIGH_UNCERTAINTY)
    - Reason codes for every decision (explainability)
    - Configurable risk parameter λ for aggressiveness tuning
    - Expected cost computation over the KPT distribution
    - Monte Carlo offline parameter optimization

Dispatch optimization objective:
    Minimize: E[rider_wait_cost] + λ * E[late_penalty]

    The safety buffer absorbs uncertainty:
        safety_buffer = a * σ_KPT + b * KVI + c * (1 - MRI) - supply_adj

    Assignment delay:
        assign_delay = max(0, E[KPT] - travel_time - safety_buffer)

Production deployment:
    - Latency target: P99 < 50ms (most critical path)
    - Called for every new order — must be fast
    - Horizontally scaled, no shared state between instances
    - Circuit breaker to kitchen_state and reliability services
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.models import (
    DispatchDecision,
    MerchantFeatures,
)
from core.optimization import (
    compute_congestion_with_memory,
    compute_expected_cost,
    compute_kpt_distribution,
    compute_kvi_full,
    compute_mri_full,
    make_dispatch_decision,
    monte_carlo_optimize_lambda,
)

router = APIRouter(prefix="/dispatch", tags=["Dispatch Control"])

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
# Request / Response Models
# ---------------------------------------------------------------------------


class DispatchRequest(BaseModel):
    """Inbound dispatch request for a single order."""

    order_id: str
    merchant_id: str
    expected_travel_time_minutes: float = Field(
        description="Estimated rider travel time to merchant in minutes"
    )
    nearby_rider_count: int | None = Field(
        None,
        description="Override nearby rider count. If None, uses feature store value.",
    )
    risk_lambda_override: float | None = Field(
        None,
        description="Override the global risk parameter λ for this request.",
    )


class BatchDispatchRequest(BaseModel):
    """Batch dispatch request for multiple orders."""

    requests: list[DispatchRequest]


class DispatchResponse(BaseModel):
    """Extended dispatch response with cost analysis."""

    decision: DispatchDecision
    expected_cost: float = Field(
        description="Expected total cost (rider wait + λ * late penalty)"
    )
    alternative_costs: dict[str, float] = Field(
        default_factory=dict,
        description="Cost under alternative risk parameters for comparison",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/decide",
    response_model=DispatchResponse,
    summary="Make an uncertainty-aware dispatch decision",
    description=(
        "Orchestrates kitchen state estimation, reliability scoring, and KPT "
        "distribution to produce an optimal dispatch decision with full "
        "explainability. Returns safety buffer breakdown, risk classification, "
        "and reason codes."
    ),
)
async def dispatch_decide(request: DispatchRequest) -> DispatchResponse:
    """Full dispatch decision pipeline.

    Execution flow:
        1. Fetch features from feature store
        2. Compute kitchen congestion state (with memory)
        3. Compute KPT distribution (LogNormal)
        4. Compute MRI (composite sigmoid)
        5. Compute KVI (scaled volatility)
        6. Build safety buffer (supply-aware)
        7. Determine assignment delay
        8. Classify risk level
        9. Generate reason codes
        10. Compute expected cost over KPT distribution

    In production, steps 2-5 could be parallelized via async tasks
    since they read from the same feature store independently.
    """
    store = _get_store()
    config = _get_config()
    now = datetime.now()

    # 1. Resolve features
    features = store.get_merchant_features(request.merchant_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No features for merchant {request.merchant_id}. "
            "Run simulation or process events first.",
        )

    # Apply override for nearby rider count (supply-awareness)
    if request.nearby_rider_count is not None:
        features = features.model_copy(
            update={"nearby_rider_count": request.nearby_rider_count}
        )

    # Apply risk lambda override if provided
    effective_config = config
    if request.risk_lambda_override is not None:
        effective_config = config.model_copy(deep=True)
        effective_config.dispatch.risk_lambda = request.risk_lambda_override

    # 2. Kitchen state (stateful)
    prev_congestion = store.get_previous_congestion(request.merchant_id)
    kitchen_state = compute_congestion_with_memory(features, prev_congestion, effective_config)
    store.set_previous_congestion(request.merchant_id, kitchen_state.congestion_score)

    # 3. KPT distribution
    kpt_dist = compute_kpt_distribution(
        kitchen_state.congestion_score, request.merchant_id, now, effective_config
    )

    # 4. MRI
    sample_count = int(store.get(request.merchant_id, "for_sample_count", default=10))
    mri = compute_mri_full(features, sample_count, effective_config)

    # 5. KVI
    kvi = compute_kvi_full(features)

    # 6-9. Dispatch decision
    decision = make_dispatch_decision(
        order_id=request.order_id,
        merchant_id=request.merchant_id,
        features=features,
        kitchen_state=kitchen_state,
        kpt_dist=kpt_dist,
        mri=mri,
        kvi=kvi,
        expected_travel_time=request.expected_travel_time_minutes,
        config=effective_config,
    )

    # 10. Expected cost
    expected_cost = compute_expected_cost(
        kpt_distribution=kpt_dist,
        travel_time=request.expected_travel_time_minutes,
        assign_delay=decision.assign_delay_minutes,
        risk_lambda=effective_config.dispatch.risk_lambda,
    )

    # Alternative cost analysis at different λ values
    alternative_costs = {}
    for alt_lambda in [0.5, 1.0, 2.0, 3.0]:
        if alt_lambda != effective_config.dispatch.risk_lambda:
            alt_cost = compute_expected_cost(
                kpt_distribution=kpt_dist,
                travel_time=request.expected_travel_time_minutes,
                assign_delay=decision.assign_delay_minutes,
                risk_lambda=alt_lambda,
            )
            alternative_costs[f"lambda_{alt_lambda}"] = round(alt_cost, 4)

    return DispatchResponse(
        decision=decision,
        expected_cost=round(expected_cost, 4),
        alternative_costs=alternative_costs,
    )


@router.post(
    "/batch",
    response_model=list[DispatchResponse],
    summary="Batch dispatch decisions for multiple orders",
    description="Process multiple dispatch requests in a single call for efficiency.",
)
async def batch_dispatch(batch: BatchDispatchRequest) -> list[DispatchResponse]:
    """Process multiple dispatch requests.

    In production, this would be parallelized across worker threads.
    Useful during order batching windows (e.g., every 15 seconds).
    """
    results = []
    for request in batch.requests:
        try:
            result = await dispatch_decide(request)
            results.append(result)
        except HTTPException:
            # Skip merchants without features in batch mode
            continue
    return results


@router.post(
    "/optimize-lambda",
    summary="Run Monte Carlo simulation to find optimal λ",
    description=(
        "Offline simulation varying the risk parameter λ to find the value "
        "that minimizes weighted cost. Uses the current merchant's KPT "
        "distribution parameters."
    ),
)
async def optimize_lambda(
    merchant_id: str,
    travel_time_minutes: float = 5.0,
    n_samples: int = 10_000,
) -> dict:
    """Find optimal λ for a merchant cluster via Monte Carlo simulation.

    This would be run offline (e.g., nightly batch job) to tune
    parameters per merchant cluster before deployment.
    """
    store = _get_store()
    config = _get_config()
    now = datetime.now()

    features = store.get_merchant_features(merchant_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"No features for {merchant_id}")

    prev_congestion = store.get_previous_congestion(merchant_id)
    kitchen_state = compute_congestion_with_memory(features, prev_congestion, config)
    kpt_dist = compute_kpt_distribution(
        kitchen_state.congestion_score, merchant_id, now, config
    )

    result = monte_carlo_optimize_lambda(
        kpt_mu=kpt_dist.mu,
        kpt_sigma=kpt_dist.sigma,
        travel_time=travel_time_minutes,
        safety_buffer=2.0,  # Nominal buffer for simulation
        n_samples=n_samples,
    )

    return {
        "merchant_id": merchant_id,
        "congestion_score": kitchen_state.congestion_score,
        "kpt_mean": kpt_dist.mean_minutes,
        "kpt_std": kpt_dist.std_minutes,
        **result,
    }
