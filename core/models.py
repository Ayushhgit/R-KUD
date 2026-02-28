"""
Pydantic data contracts for RPKOE.

All inter-service communication uses these strongly-typed models.
Each model documents its production semantics, including fields needed
for idempotency, ordering, and traceability.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Classification of inbound event stream messages."""

    ORDER_PLACED = "ORDER_PLACED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_PREPARED = "ORDER_PREPARED"  # Food-On-Rack (FOR) signal
    RIDER_ASSIGNED = "RIDER_ASSIGNED"
    RIDER_ARRIVED = "RIDER_ARRIVED"
    RIDER_PICKED_UP = "RIDER_PICKED_UP"
    ORDER_DELIVERED = "ORDER_DELIVERED"
    ORDER_CANCELLED = "ORDER_CANCELLED"


class RiskLevel(str, Enum):
    """Dispatch risk classification for explainability."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"


class AlertSeverity(str, Enum):
    """Severity levels for monitoring alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Inbound Event Schemas
# ---------------------------------------------------------------------------

class OrderEvent(BaseModel):
    """Represents an order lifecycle event from the event stream.

    In production, these arrive via Kafka partitioned by merchant_id.
    event_id enables idempotent processing; duplicate events with the
    same event_id are discarded by the feature aggregator.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    order_id: str
    merchant_id: str
    timestamp: datetime
    estimated_prep_time_minutes: Optional[float] = None
    actual_prep_time_minutes: Optional[float] = None

    # Merchant-provided estimate at FOR marking
    merchant_estimated_kpt_minutes: Optional[float] = None


class RiderEvent(BaseModel):
    """Rider telemetry and lifecycle event.

    Late-arriving rider events (timestamp < last_processed) are flagged
    and handled via the feature aggregator's ordering logic.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    rider_id: str
    order_id: str
    merchant_id: str
    timestamp: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    estimated_travel_time_minutes: Optional[float] = None


class FOREvent(BaseModel):
    """Food-On-Rack signal — the merchant's noisy marking event.

    This is the primary signal for kitchen state estimation.
    marking_delay = (actual_ready_time - for_marking_time) measures
    how late (positive) or early (negative) the merchant's mark was.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id: str
    order_id: str
    timestamp: datetime
    marking_delay_minutes: float = Field(
        description="Delay between FOR marking and actual readiness. "
        "Positive = food was ready AFTER mark, Negative = food was ready BEFORE mark."
    )


# ---------------------------------------------------------------------------
# Feature / State Models
# ---------------------------------------------------------------------------

class MerchantFeatures(BaseModel):
    """Aggregated features for a single merchant over sliding windows.

    Produced by the feature aggregator and consumed by all downstream services.
    """

    merchant_id: str
    timestamp: datetime

    # Order volume
    active_orders: int = 0
    orders_last_5min: int = 0
    orders_last_15min: int = 0

    # Throughput
    max_capacity: int = 15
    throughput_saturation: float = Field(
        0.0, description="active_orders / max_capacity, clamped to [0, 1]"
    )

    # Residual drift: rolling mean of (actual_prep - estimated_prep)
    residual_drift_minutes: float = 0.0

    # Time-of-day factor: normalized demand multiplier (0.0 = off-peak, 1.0 = peak)
    time_of_day_factor: float = 0.0

    # FOR marking statistics
    marking_std: float = Field(
        0.5, description="Std dev of marking_delay across recent observations"
    )
    mark_on_arrival_rate: float = Field(
        0.5, description="Fraction of orders where FOR was marked at or before actual ready"
    )
    delay_entropy: float = Field(
        1.0, description="Shannon entropy of discretized marking delays"
    )

    # Pickup delays
    rolling_pickup_delay_std: float = Field(
        0.0, description="Rolling std of pickup delay over recent window"
    )
    external_rush_factor: float = Field(
        1.0, description="Multiplier for external demand surges (events, weather)"
    )

    # Supply context
    nearby_rider_count: int = Field(
        5, description="Number of available riders within pickup radius"
    )

    # Partition metadata
    partition_id: int = 0


class KitchenState(BaseModel):
    """Estimated hidden kitchen congestion state for a merchant."""

    merchant_id: str
    timestamp: datetime
    congestion_score: float = Field(description="Normalized congestion [0, 1+]")
    active_orders: int
    residual_drift: float
    throughput_saturation: float
    spike_detected: bool = False
    previous_congestion: Optional[float] = None


class KPTDistribution(BaseModel):
    """Probabilistic Kitchen Preparation Time distribution.

    KPT ~ LogNormal(μ, σ) where μ and σ are functions of congestion_score.
    """

    merchant_id: str
    timestamp: datetime
    mu: float = Field(description="Log-mean of the LogNormal distribution")
    sigma: float = Field(description="Log-std of the LogNormal distribution")
    mean_minutes: float = Field(description="E[KPT] in minutes")
    std_minutes: float = Field(description="Std[KPT] in minutes")
    p50_minutes: float = Field(description="Median KPT")
    p75_minutes: float = Field(description="75th percentile KPT")
    p90_minutes: float = Field(description="90th percentile KPT")
    p95_minutes: float = Field(description="95th percentile KPT")


class MerchantReliability(BaseModel):
    """Merchant Reliability Index (MRI) — composite score.

    Composite MRI = sigmoid(w1 * inv_std + w2 * mark_on_arrival_rate + w3 * inv_entropy)
    Ranges in (0, 1), where higher = more reliable.
    """

    merchant_id: str
    timestamp: datetime
    mri_score: float = Field(description="Composite MRI in (0, 1)")
    marking_std: float
    mark_on_arrival_rate: float
    delay_entropy: float
    sample_count: int = Field(description="Number of observations in the window")


class KitchenVolatility(BaseModel):
    """Kitchen Volatility Index (KVI).

    KVI = rolling_std(pickup_delay) * external_rush_factor
    """

    merchant_id: str
    timestamp: datetime
    kvi_score: float
    rolling_std_pickup_delay: float
    external_rush_factor: float


class SafetyBufferBreakdown(BaseModel):
    """Decomposed safety buffer for dispatch explainability."""

    kpt_std_component: float = Field(description="a * kpt_std")
    kvi_component: float = Field(description="b * KVI")
    mri_component: float = Field(description="c * (1 - MRI)")
    supply_adjustment: float = Field(
        0.0, description="Reduction applied when rider supply is low"
    )
    total: float


class DispatchDecision(BaseModel):
    """Structured dispatch decision with full explainability.

    Returned by the dispatch service for each order.
    """

    order_id: str
    merchant_id: str
    timestamp: datetime

    # Timing
    predicted_kpt_minutes: float
    expected_travel_time_minutes: float
    safety_buffer: SafetyBufferBreakdown
    assign_delay_minutes: float = Field(
        description="How long to delay rider assignment (max(0, kpt - travel - buffer))"
    )

    # Risk
    risk_level: RiskLevel
    risk_lambda: float

    # Context
    congestion_score: float
    kpt_std_minutes: float
    mri_score: float
    kvi_score: float
    nearby_rider_count: int

    # Explainability
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the dispatch decision",
    )


# ---------------------------------------------------------------------------
# Monitoring Models
# ---------------------------------------------------------------------------

class MonitoringAlert(BaseModel):
    """A single monitoring alert from the guardrail system."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    severity: AlertSeverity
    metric_name: str
    observed_value: float
    threshold_value: float
    message: str


class CalibrationMetrics(BaseModel):
    """Distribution calibration monitoring metrics."""

    observed_coverage_p90: float = Field(
        description="Fraction of actuals that fell inside predicted P90 interval"
    )
    expected_coverage_p90: float = 0.90
    calibration_error: float = Field(
        description="observed_coverage - expected_coverage (positive = overconfident)"
    )
    is_calibrated: bool = True


class SystemMetrics(BaseModel):
    """Aggregate system health metrics for the monitoring dashboard."""

    timestamp: datetime

    # Rider wait
    rider_wait_p50_minutes: float = 0.0
    rider_wait_p90_minutes: float = 0.0
    rider_wait_mean_minutes: float = 0.0

    # ETA accuracy
    eta_p90_minutes: float = 0.0
    eta_mean_absolute_error_minutes: float = 0.0

    # Cancellation
    cancellation_rate: float = 0.0
    cancellation_rate_change_pct: float = 0.0

    # Calibration
    calibration: Optional[CalibrationMetrics] = None

    # Congestion drift
    mean_congestion_prediction_error: float = 0.0

    # MRI health
    mean_mri_score: float = 0.0
    mri_degradation_detected: bool = False

    # Volume
    total_orders_processed: int = 0
    total_dispatch_decisions: int = 0

    # Alerts
    active_alerts: list[MonitoringAlert] = Field(default_factory=list)
