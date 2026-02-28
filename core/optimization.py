"""
Pure mathematical functions for RPKOE.

This module contains ALL computation logic — stateless, side-effect-free
functions that transform inputs into outputs. No I/O, no state mutation,
no service calls. This makes every function independently testable and
reusable across services.

Mathematical foundations:
    - Congestion: weighted linear combination with exponential decay memory
    - KPT: LogNormal distribution parameterized by congestion score
    - MRI: Composite sigmoid over marking quality features
    - KVI: Scaled rolling volatility with rush-factor amplification
    - Dispatch: Uncertainty-aware cost minimization with supply adjustment
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from core.config import RPKOEConfig
from core.models import (
    DispatchDecision,
    KitchenState,
    KitchenVolatility,
    KPTDistribution,
    MerchantFeatures,
    MerchantReliability,
    RiskLevel,
    SafetyBufferBreakdown,
)
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Kitchen Congestion Estimation
# ---------------------------------------------------------------------------


def compute_instantaneous_congestion(
    features: MerchantFeatures,
    config: RPKOEConfig,
) -> float:
    """Compute the instantaneous (memoryless) congestion score.

    congestion_inst = α * (active_orders / max_capacity)
                    + β * residual_drift_norm
                    + γ * time_of_day_factor
                    + δ * throughput_saturation

    All inputs are normalized to roughly [0, 1] before weighting.
    """
    c = config.congestion

    # Normalize active orders to [0, 1]
    active_norm = min(features.active_orders / max(features.max_capacity, 1), 1.0)

    # Normalize residual drift: clamp to [-5, 5] then map to [0, 1]
    drift_clamped = max(-5.0, min(features.residual_drift_minutes, 5.0))
    drift_norm = (drift_clamped + 5.0) / 10.0

    score = (
        c.alpha * active_norm
        + c.beta * drift_norm
        + c.gamma * features.time_of_day_factor
        + c.delta * features.throughput_saturation
    )
    return max(0.0, score)


def compute_congestion_with_memory(
    features: MerchantFeatures,
    previous_congestion: Optional[float],
    config: RPKOEConfig,
) -> KitchenState:
    """Stateful congestion estimation with exponential decay memory.

    congestion_t = decay * congestion_(t-1) + (1 - decay) * instantaneous_score

    Also applies a spike penalty when residual drift exceeds threshold,
    indicating a sudden kitchen backup that simple averaging would smooth out.

    Args:
        features: Current aggregated merchant features.
        previous_congestion: Last computed congestion (None for cold-start).
        config: System configuration.

    Returns:
        KitchenState with the updated congestion score.
    """
    c = config.congestion
    instantaneous = compute_instantaneous_congestion(features, config)

    if previous_congestion is not None:
        congestion = c.decay_factor * previous_congestion + (1 - c.decay_factor) * instantaneous
    else:
        # Cold start: use instantaneous score directly
        congestion = instantaneous

    # Spike detection
    spike_detected = abs(features.residual_drift_minutes) > c.spike_threshold
    if spike_detected:
        congestion += c.spike_penalty

    return KitchenState(
        merchant_id=features.merchant_id,
        timestamp=features.timestamp,
        congestion_score=max(0.0, min(congestion, 2.0)),  # Soft clamp
        active_orders=features.active_orders,
        residual_drift=features.residual_drift_minutes,
        throughput_saturation=features.throughput_saturation,
        spike_detected=spike_detected,
        previous_congestion=previous_congestion,
    )


# ---------------------------------------------------------------------------
# 2. KPT Distribution (LogNormal)
# ---------------------------------------------------------------------------


def compute_kpt_distribution(
    congestion_score: float,
    merchant_id: str,
    timestamp: datetime,
    config: RPKOEConfig,
) -> KPTDistribution:
    """Compute the probabilistic KPT distribution.

    KPT ~ LogNormal(μ, σ) where:
        μ = mu_base + mu_congestion_scale * congestion_score
        σ = sigma_base + sigma_congestion_scale * congestion_score

    The LogNormal is a natural fit for preparation times because:
        - Times are strictly positive
        - Right-skewed (occasional long waits)
        - Mode is less than mean (most orders are faster than average)

    Returns:
        KPTDistribution with μ, σ, and percentiles in minutes.
    """
    k = config.kpt
    mu = k.mu_base + k.mu_congestion_scale * congestion_score
    sigma = max(k.sigma_base + k.sigma_congestion_scale * congestion_score, 0.05)

    # LogNormal moments: E[X] = exp(μ + σ²/2), Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
    mean = math.exp(mu + sigma**2 / 2)
    variance = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
    std = math.sqrt(variance)

    # Percentiles via inverse CDF
    dist = sp_stats.lognorm(s=sigma, scale=math.exp(mu))

    return KPTDistribution(
        merchant_id=merchant_id,
        timestamp=timestamp,
        mu=mu,
        sigma=sigma,
        mean_minutes=round(mean, 2),
        std_minutes=round(std, 2),
        p50_minutes=round(float(dist.ppf(0.50)), 2),
        p75_minutes=round(float(dist.ppf(0.75)), 2),
        p90_minutes=round(float(dist.ppf(0.90)), 2),
        p95_minutes=round(float(dist.ppf(0.95)), 2),
    )


# ---------------------------------------------------------------------------
# 3. Merchant Reliability Index (MRI)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def compute_mri(
    marking_std: float,
    mark_on_arrival_rate: float,
    delay_entropy: float,
    config: RPKOEConfig,
) -> float:
    """Compute the composite Merchant Reliability Index.

    MRI = sigmoid(w1 * (1 / (1 + marking_std))
                + w2 * mark_on_arrival_rate
                + w3 * (1 / (1 + delay_entropy)))

    Interpretation:
        - Low marking_std → consistent FOR marking → reliable
        - High mark_on_arrival_rate → marks food as ready accurately → reliable
        - Low delay_entropy → predictable delay pattern → reliable

    Returns:
        MRI score in (0, 1).
    """
    m = config.mri
    inv_std = 1.0 / (1.0 + max(marking_std, 0.0))
    inv_entropy = 1.0 / (1.0 + max(delay_entropy, 0.0))

    linear = m.w1 * inv_std + m.w2 * mark_on_arrival_rate + m.w3 * inv_entropy
    return round(_sigmoid(linear), 4)


def compute_mri_full(
    features: MerchantFeatures,
    sample_count: int,
    config: RPKOEConfig,
) -> MerchantReliability:
    """Compute full MRI with all metadata."""
    mri_score = compute_mri(
        features.marking_std,
        features.mark_on_arrival_rate,
        features.delay_entropy,
        config,
    )
    return MerchantReliability(
        merchant_id=features.merchant_id,
        timestamp=features.timestamp,
        mri_score=mri_score,
        marking_std=features.marking_std,
        mark_on_arrival_rate=features.mark_on_arrival_rate,
        delay_entropy=features.delay_entropy,
        sample_count=sample_count,
    )


# ---------------------------------------------------------------------------
# 4. Kitchen Volatility Index (KVI)
# ---------------------------------------------------------------------------


def compute_kvi(
    rolling_std_pickup_delay: float,
    external_rush_factor: float,
) -> float:
    """Compute Kitchen Volatility Index.

    KVI = rolling_std(pickup_delay) * external_rush_factor

    Higher KVI means the kitchen's output timing is unpredictable,
    and external factors (events, weather, festivals) amplify the risk.
    """
    return round(max(rolling_std_pickup_delay, 0.0) * max(external_rush_factor, 0.0), 4)


def compute_kvi_full(
    features: MerchantFeatures,
) -> KitchenVolatility:
    """Compute full KVI with metadata."""
    kvi = compute_kvi(features.rolling_pickup_delay_std, features.external_rush_factor)
    return KitchenVolatility(
        merchant_id=features.merchant_id,
        timestamp=features.timestamp,
        kvi_score=kvi,
        rolling_std_pickup_delay=features.rolling_pickup_delay_std,
        external_rush_factor=features.external_rush_factor,
    )


# ---------------------------------------------------------------------------
# 5. Dispatch Optimization
# ---------------------------------------------------------------------------


def compute_safety_buffer(
    kpt_std: float,
    kvi: float,
    mri: float,
    nearby_rider_count: int,
    config: RPKOEConfig,
) -> SafetyBufferBreakdown:
    """Compute the adaptive dispatch safety buffer.

    safety_buffer = a * kpt_std + b * KVI + c * (1 - MRI) - supply_adjustment

    The supply adjustment reduces the buffer when rider supply is low,
    accepting slightly more risk to avoid stranding orders without riders.

    Returns:
        SafetyBufferBreakdown with component-level decomposition.
    """
    s = config.safety
    d = config.dispatch

    kpt_component = s.a * kpt_std
    kvi_component = s.b * kvi
    mri_component = s.c * (1.0 - mri)

    total = kpt_component + kvi_component + mri_component

    # Supply-aware adjustment: if rider supply is low, reduce buffer
    supply_adjustment = 0.0
    if nearby_rider_count < d.low_supply_threshold:
        supply_adjustment = total * d.low_supply_buffer_reduction
        total = total * (1.0 - d.low_supply_buffer_reduction)

    return SafetyBufferBreakdown(
        kpt_std_component=round(kpt_component, 3),
        kvi_component=round(kvi_component, 3),
        mri_component=round(mri_component, 3),
        supply_adjustment=round(supply_adjustment, 3),
        total=round(max(total, 0.0), 3),
    )


def compute_assign_delay(
    predicted_kpt: float,
    expected_travel_time: float,
    safety_buffer: float,
) -> float:
    """Compute optimal rider assignment delay.

    assign_delay = max(0, predicted_kpt - expected_travel_time - safety_buffer)

    If predicted KPT is much longer than travel time, delay dispatch
    so the rider arrives closer to when food is actually ready.
    """
    delay = predicted_kpt - expected_travel_time - safety_buffer
    return round(max(0.0, delay), 2)


def compute_dispatch_cost(
    rider_wait_minutes: float,
    is_late: bool,
    risk_lambda: float,
) -> float:
    """Compute the dispatch cost objective.

    cost = rider_wait_cost + λ * late_penalty

    Where:
        rider_wait_cost = rider_wait_minutes  (linear)
        late_penalty = 1.0 if rider arrived after food ready, 0.0 otherwise

    In a full production system, these would be:
        E[rider_wait_cost] + λ * P[late_arrival] * late_penalty_magnitude
    computed over the KPT distribution.
    """
    late_penalty = 1.0 if is_late else 0.0
    return rider_wait_minutes + risk_lambda * late_penalty


def compute_expected_cost(
    kpt_distribution: KPTDistribution,
    travel_time: float,
    assign_delay: float,
    risk_lambda: float,
) -> float:
    """Compute expected cost over the KPT distribution.

    E[cost] = E[max(0, arrival_time - kpt)] + λ * P[arrival > kpt]

    Where arrival_time = assign_delay + travel_time.

    Uses the LogNormal CDF for the probability calculation.
    """
    arrival_time = assign_delay + travel_time

    # P[KPT < arrival_time] = probability rider arrives before food is ready (rider waits)
    # P[KPT > arrival_time] = probability food is ready before rider (late arrival)
    dist = sp_stats.lognorm(s=kpt_distribution.sigma, scale=math.exp(kpt_distribution.mu))

    p_late = 1.0 - dist.cdf(arrival_time)  # P[KPT > arrival_time]

    # Expected rider wait: E[max(0, arrival_time - KPT)]
    # = arrival_time * P[KPT < arrival] - E[KPT * I(KPT < arrival)]
    # Approximation using distribution moments:
    expected_wait = max(0.0, arrival_time - kpt_distribution.mean_minutes)

    return expected_wait + risk_lambda * p_late


def make_dispatch_decision(
    order_id: str,
    merchant_id: str,
    features: MerchantFeatures,
    kitchen_state: KitchenState,
    kpt_dist: KPTDistribution,
    mri: MerchantReliability,
    kvi: KitchenVolatility,
    expected_travel_time: float,
    config: RPKOEConfig,
) -> DispatchDecision:
    """Orchestrate a full dispatch decision with explainability.

    This is the top-level dispatch function that:
    1. Computes the safety buffer from all risk signals
    2. Determines the optimal assignment delay
    3. Classifies risk level
    4. Generates human-readable reason codes

    Returns:
        DispatchDecision with complete breakdown.
    """
    d = config.dispatch

    # 1. Safety buffer
    buffer = compute_safety_buffer(
        kpt_std=kpt_dist.std_minutes,
        kvi=kvi.kvi_score,
        mri=mri.mri_score,
        nearby_rider_count=features.nearby_rider_count,
        config=config,
    )

    # 2. Assignment delay
    assign_delay = compute_assign_delay(
        predicted_kpt=kpt_dist.mean_minutes,
        expected_travel_time=expected_travel_time,
        safety_buffer=buffer.total,
    )

    # 3. Risk classification
    if buffer.total > d.high_uncertainty_threshold:
        risk_level = RiskLevel.HIGH_UNCERTAINTY
    elif buffer.total > d.moderate_uncertainty_threshold:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    # 4. Reason codes for explainability
    reasons: list[str] = []

    if kitchen_state.spike_detected:
        reasons.append("SPIKE_DETECTED: Residual drift exceeds threshold")
    if kitchen_state.congestion_score > 0.7:
        reasons.append(f"HIGH_CONGESTION: Score {kitchen_state.congestion_score:.2f}")
    if mri.mri_score < 0.5:
        reasons.append(f"LOW_RELIABILITY: MRI {mri.mri_score:.2f}")
    if kvi.kvi_score > 2.0:
        reasons.append(f"HIGH_VOLATILITY: KVI {kvi.kvi_score:.2f}")
    if features.nearby_rider_count < d.low_supply_threshold:
        reasons.append(
            f"LOW_SUPPLY: {features.nearby_rider_count} riders nearby, "
            f"buffer reduced by {d.low_supply_buffer_reduction:.0%}"
        )
    if assign_delay > 0:
        reasons.append(f"DELAYED_DISPATCH: {assign_delay:.1f}min delay applied")
    if not reasons:
        reasons.append("NORMAL: All signals within acceptable range")

    return DispatchDecision(
        order_id=order_id,
        merchant_id=merchant_id,
        timestamp=features.timestamp,
        predicted_kpt_minutes=kpt_dist.mean_minutes,
        expected_travel_time_minutes=expected_travel_time,
        safety_buffer=buffer,
        assign_delay_minutes=assign_delay,
        risk_level=risk_level,
        risk_lambda=d.risk_lambda,
        congestion_score=kitchen_state.congestion_score,
        kpt_std_minutes=kpt_dist.std_minutes,
        mri_score=mri.mri_score,
        kvi_score=kvi.kvi_score,
        nearby_rider_count=features.nearby_rider_count,
        reason_codes=reasons,
    )


# ---------------------------------------------------------------------------
# 6. Monte Carlo Simulation for Offline Parameter Optimization
# ---------------------------------------------------------------------------


def monte_carlo_optimize_lambda(
    kpt_mu: float,
    kpt_sigma: float,
    travel_time: float,
    safety_buffer: float,
    lambda_candidates: list[float] | None = None,
    n_samples: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Offline Monte Carlo simulation to find optimal risk parameter λ.

    For each λ candidate, simulates n_samples KPT draws from the LogNormal
    distribution and computes the expected cost:
        E[max(0, arrival - KPT)] + λ * E[I(KPT > arrival)]

    This enables per-merchant-cluster parameter tuning before deployment.

    Returns:
        Dict with optimal λ, costs per candidate, and simulation metadata.
    """
    if lambda_candidates is None:
        lambda_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    rng = np.random.default_rng(rng_seed)
    kpt_samples = rng.lognormal(mean=kpt_mu, sigma=kpt_sigma, size=n_samples)

    results = {}
    for lam in lambda_candidates:
        assign_delay = max(0.0, float(np.median(kpt_samples)) - travel_time - safety_buffer)
        arrival_time = assign_delay + travel_time

        rider_waits = np.maximum(0.0, arrival_time - kpt_samples)
        late_flags = (kpt_samples > arrival_time).astype(float)

        cost = float(np.mean(rider_waits) + lam * np.mean(late_flags))
        results[lam] = {
            "expected_cost": round(cost, 4),
            "mean_rider_wait": round(float(np.mean(rider_waits)), 4),
            "late_probability": round(float(np.mean(late_flags)), 4),
            "assign_delay": round(assign_delay, 4),
        }

    optimal_lambda = min(results, key=lambda k: results[k]["expected_cost"])
    return {
        "optimal_lambda": optimal_lambda,
        "results_by_lambda": results,
        "n_samples": n_samples,
        "kpt_mu": kpt_mu,
        "kpt_sigma": kpt_sigma,
    }
