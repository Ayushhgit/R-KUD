"""
Monitoring & Guardrails Service.

Tracks system health metrics and fires alerts when guardrail thresholds
are breached. Provides operational visibility into:
    - Rider wait time distribution
    - ETA accuracy
    - Distribution calibration health
    - Congestion prediction drift
    - MRI degradation
    - Cancellation rate spikes

Guardrail actions (production):
    - ETA P90 > baseline + 2min → PagerDuty alert to ops
    - Rider wait P90 spike > 20% → auto-reduce safety buffer by 10%
    - Cancellation rate increase → rollback adaptive dispatch to safe defaults
    - Calibration error > threshold → trigger model recalibration pipeline
    - MRI degradation > 15% → flag merchant for manual review

Production deployment:
    - Metrics exported to Prometheus / Datadog
    - Grafana dashboards for real-time visibility
    - PagerDuty integration for critical alerts
    - This service is the observability layer, not the action layer
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter

from core.config import RPKOEConfig
from core.models import AlertSeverity, CalibrationMetrics, MonitoringAlert, SystemMetrics

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Guardrails"])

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_config: RPKOEConfig | None = None


def set_dependencies(config: RPKOEConfig) -> None:
    global _config
    _config = config


def _get_config() -> RPKOEConfig:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


# ---------------------------------------------------------------------------
# In-memory metrics storage (production: Prometheus / time-series DB)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Centralized metrics collection with windowed aggregation.

    In production, this would be replaced by Prometheus client libraries
    (Counter, Histogram, Gauge) pushing to a time-series database.
    """

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size

        # Rider wait observations: deque of (timestamp, wait_minutes)
        self.rider_waits: deque[tuple[datetime, float]] = deque(maxlen=window_size)

        # ETA observations: deque of (timestamp, predicted, actual)
        self.eta_observations: deque[tuple[datetime, float, float]] = deque(maxlen=window_size)

        # Cancellation tracking
        self.total_orders: int = 0
        self.total_cancellations: int = 0
        self._prev_cancellation_rate: float = 0.0

        # Congestion prediction tracking
        self.congestion_predictions: deque[tuple[datetime, float, float]] = deque(maxlen=window_size)

        # MRI tracking per merchant
        self.mri_history: dict[str, deque[tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Calibration per merchant
        self.calibration_observations: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=200)
        )

        # Active alerts
        self.active_alerts: list[MonitoringAlert] = []

        # Dispatch decision counter
        self.total_dispatch_decisions: int = 0

    def record_rider_wait(self, wait_minutes: float) -> None:
        """Record a rider wait time observation."""
        self.rider_waits.append((datetime.now(), wait_minutes))

    def record_eta(self, predicted_minutes: float, actual_minutes: float) -> None:
        """Record an ETA prediction vs actual observation."""
        self.eta_observations.append((datetime.now(), predicted_minutes, actual_minutes))

    def record_order(self, cancelled: bool = False) -> None:
        """Record an order outcome."""
        self.total_orders += 1
        if cancelled:
            self.total_cancellations += 1

    def record_congestion_prediction(
        self, predicted: float, actual: float
    ) -> None:
        """Record congestion prediction vs actual for drift monitoring."""
        self.congestion_predictions.append((datetime.now(), predicted, actual))

    def record_mri(self, merchant_id: str, mri_score: float) -> None:
        """Track MRI over time for degradation detection."""
        self.mri_history[merchant_id].append((datetime.now(), mri_score))

    def record_calibration(
        self, merchant_id: str, predicted_p90: float, actual: float
    ) -> None:
        """Record for distribution calibration monitoring."""
        self.calibration_observations[merchant_id].append((predicted_p90, actual))

    def record_dispatch(self) -> None:
        """Increment dispatch decision counter."""
        self.total_dispatch_decisions += 1


# Singleton collector instance
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """Access the metrics collector (dependency injection point)."""
    return _collector


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_rider_wait_metrics(
    collector: MetricsCollector,
) -> tuple[float, float, float]:
    """Compute P50, P90, and mean rider wait times."""
    if not collector.rider_waits:
        return 0.0, 0.0, 0.0

    waits = [w for _, w in collector.rider_waits]
    return (
        round(float(np.percentile(waits, 50)), 2),
        round(float(np.percentile(waits, 90)), 2),
        round(float(np.mean(waits)), 2),
    )


def _compute_eta_metrics(
    collector: MetricsCollector,
) -> tuple[float, float]:
    """Compute ETA P90 and mean absolute error."""
    if not collector.eta_observations:
        return 0.0, 0.0

    actuals = [a for _, _, a in collector.eta_observations]
    errors = [abs(p - a) for _, p, a in collector.eta_observations]

    return (
        round(float(np.percentile(actuals, 90)), 2),
        round(float(np.mean(errors)), 2),
    )


def _compute_calibration(
    collector: MetricsCollector,
    config: RPKOEConfig,
) -> Optional[CalibrationMetrics]:
    """Compute aggregate calibration metrics across all merchants."""
    all_obs = []
    for obs in collector.calibration_observations.values():
        all_obs.extend(obs)

    if len(all_obs) < 20:
        return None

    inside_count = sum(1 for p90, actual in all_obs if actual <= p90)
    observed_coverage = inside_count / len(all_obs)
    calibration_error = observed_coverage - config.kpt.expected_coverage_p90

    return CalibrationMetrics(
        observed_coverage_p90=round(observed_coverage, 4),
        expected_coverage_p90=config.kpt.expected_coverage_p90,
        calibration_error=round(calibration_error, 4),
        is_calibrated=abs(calibration_error) <= config.kpt.calibration_alert_threshold,
    )


def _check_mri_degradation(
    collector: MetricsCollector,
    config: RPKOEConfig,
) -> tuple[float, bool]:
    """Check for MRI degradation across merchants."""
    if not collector.mri_history:
        return 0.0, False

    current_scores = []
    for merchant_id, history in collector.mri_history.items():
        if history:
            current_scores.append(history[-1][1])

    if not current_scores:
        return 0.0, False

    mean_mri = float(np.mean(current_scores))

    # Check if any merchant's MRI dropped significantly
    degraded = False
    for merchant_id, history in collector.mri_history.items():
        if len(history) >= 10:
            recent = [s for _, s in list(history)[-5:]]
            older = [s for _, s in list(history)[-10:-5]]
            if older and recent:
                drop = float(np.mean(older)) - float(np.mean(recent))
                if drop > config.monitoring.mri_degradation_threshold:
                    degraded = True
                    break

    return round(mean_mri, 4), degraded


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


def _generate_alerts(
    collector: MetricsCollector,
    config: RPKOEConfig,
    wait_p90: float,
    eta_p90: float,
    calibration: Optional[CalibrationMetrics],
    mri_degraded: bool,
) -> list[MonitoringAlert]:
    """Evaluate all guardrails and generate alerts."""
    alerts: list[MonitoringAlert] = []
    now = datetime.now()
    mon = config.monitoring

    # 1. Rider wait P90
    if wait_p90 > mon.rider_wait_p90_alert_minutes:
        alerts.append(
            MonitoringAlert(
                timestamp=now,
                severity=AlertSeverity.WARNING,
                metric_name="rider_wait_p90",
                observed_value=wait_p90,
                threshold_value=mon.rider_wait_p90_alert_minutes,
                message=(
                    f"Rider wait P90 ({wait_p90:.1f}min) exceeds threshold "
                    f"({mon.rider_wait_p90_alert_minutes}min). "
                    "Action: Review safety buffer parameters."
                ),
            )
        )

    # 2. ETA P90
    eta_threshold = mon.eta_p90_baseline_minutes + mon.eta_p90_alert_buffer_minutes
    if eta_p90 > eta_threshold:
        alerts.append(
            MonitoringAlert(
                timestamp=now,
                severity=AlertSeverity.WARNING,
                metric_name="eta_p90",
                observed_value=eta_p90,
                threshold_value=eta_threshold,
                message=(
                    f"ETA P90 ({eta_p90:.1f}min) exceeds baseline + buffer "
                    f"({eta_threshold:.1f}min). "
                    "Action: Investigate congestion estimation accuracy."
                ),
            )
        )

    # 3. Cancellation rate
    if collector.total_orders > 0:
        cancel_rate = collector.total_cancellations / collector.total_orders
        if cancel_rate > mon.cancellation_rate_alert_threshold:
            alerts.append(
                MonitoringAlert(
                    timestamp=now,
                    severity=AlertSeverity.CRITICAL,
                    metric_name="cancellation_rate",
                    observed_value=round(cancel_rate, 4),
                    threshold_value=mon.cancellation_rate_alert_threshold,
                    message=(
                        f"Cancellation rate ({cancel_rate:.2%}) exceeds threshold "
                        f"({mon.cancellation_rate_alert_threshold:.2%}). "
                        "Action: Consider rollback to default dispatch parameters."
                    ),
                )
            )

        # Cancellation spike detection
        if (
            collector._prev_cancellation_rate > 0
            and cancel_rate > collector._prev_cancellation_rate * (1 + mon.cancellation_spike_pct)
        ):
            alerts.append(
                MonitoringAlert(
                    timestamp=now,
                    severity=AlertSeverity.CRITICAL,
                    metric_name="cancellation_spike",
                    observed_value=round(cancel_rate, 4),
                    threshold_value=round(
                        collector._prev_cancellation_rate * (1 + mon.cancellation_spike_pct), 4
                    ),
                    message=(
                        f"Cancellation rate spiked by >{mon.cancellation_spike_pct:.0%}. "
                        "Action: ROLLBACK adaptive dispatch to safe defaults."
                    ),
                )
            )

        collector._prev_cancellation_rate = cancel_rate

    # 4. Calibration drift
    if calibration and not calibration.is_calibrated:
        alerts.append(
            MonitoringAlert(
                timestamp=now,
                severity=AlertSeverity.WARNING,
                metric_name="calibration_error",
                observed_value=calibration.calibration_error,
                threshold_value=config.kpt.calibration_alert_threshold,
                message=(
                    f"KPT distribution miscalibrated: coverage "
                    f"{calibration.observed_coverage_p90:.2%} vs expected "
                    f"{calibration.expected_coverage_p90:.2%}. "
                    "Action: Trigger model recalibration pipeline."
                ),
            )
        )

    # 5. MRI degradation
    if mri_degraded:
        alerts.append(
            MonitoringAlert(
                timestamp=now,
                severity=AlertSeverity.WARNING,
                metric_name="mri_degradation",
                observed_value=0.0,
                threshold_value=mon.mri_degradation_threshold,
                message=(
                    f"MRI degradation detected (>{mon.mri_degradation_threshold:.0%} drop). "
                    "Action: Flag affected merchants for manual review."
                ),
            )
        )

    # 6. Congestion drift
    if collector.congestion_predictions:
        errors = [abs(p - a) for _, p, a in collector.congestion_predictions]
        mean_error = float(np.mean(errors))
        if mean_error > mon.congestion_drift_alert_threshold:
            alerts.append(
                MonitoringAlert(
                    timestamp=now,
                    severity=AlertSeverity.WARNING,
                    metric_name="congestion_drift",
                    observed_value=round(mean_error, 4),
                    threshold_value=mon.congestion_drift_alert_threshold,
                    message=(
                        f"Congestion prediction drift ({mean_error:.4f}) exceeds "
                        f"threshold ({mon.congestion_drift_alert_threshold}). "
                        "Action: Review congestion model coefficients."
                    ),
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    response_model=SystemMetrics,
    summary="Get current system health metrics",
    description="Returns aggregate metrics including rider wait, ETA accuracy, "
    "calibration status, and active alerts.",
)
async def get_metrics() -> SystemMetrics:
    """Compute and return current system metrics with guardrail evaluation."""
    config = _get_config()
    collector = _collector

    # Compute metrics
    wait_p50, wait_p90, wait_mean = _compute_rider_wait_metrics(collector)
    eta_p90, eta_mae = _compute_eta_metrics(collector)
    calibration = _compute_calibration(collector, config)
    mean_mri, mri_degraded = _check_mri_degradation(collector, config)

    # Congestion drift
    cong_error = 0.0
    if collector.congestion_predictions:
        errors = [abs(p - a) for _, p, a in collector.congestion_predictions]
        cong_error = round(float(np.mean(errors)), 4)

    # Cancellation
    cancel_rate = (
        collector.total_cancellations / max(collector.total_orders, 1)
    )

    # Generate alerts
    alerts = _generate_alerts(
        collector, config, wait_p90, eta_p90, calibration, mri_degraded
    )
    collector.active_alerts = alerts

    return SystemMetrics(
        timestamp=datetime.now(),
        rider_wait_p50_minutes=wait_p50,
        rider_wait_p90_minutes=wait_p90,
        rider_wait_mean_minutes=wait_mean,
        eta_p90_minutes=eta_p90,
        eta_mean_absolute_error_minutes=eta_mae,
        cancellation_rate=round(cancel_rate, 4),
        calibration=calibration,
        mean_congestion_prediction_error=cong_error,
        mean_mri_score=mean_mri,
        mri_degradation_detected=mri_degraded,
        total_orders_processed=collector.total_orders,
        total_dispatch_decisions=collector.total_dispatch_decisions,
        active_alerts=alerts,
    )


@router.get(
    "/alerts",
    response_model=list[MonitoringAlert],
    summary="Get active guardrail alerts",
    description="Returns all currently active alerts that require attention.",
)
async def get_alerts() -> list[MonitoringAlert]:
    """Return active alerts from the last metrics evaluation."""
    return _collector.active_alerts


@router.post(
    "/record/rider-wait",
    summary="Record a rider wait observation",
)
async def record_rider_wait(wait_minutes: float) -> dict:
    """Record rider wait time for monitoring."""
    _collector.record_rider_wait(wait_minutes)
    return {"recorded": True, "wait_minutes": wait_minutes}


@router.post(
    "/record/eta",
    summary="Record an ETA prediction vs actual",
)
async def record_eta(predicted_minutes: float, actual_minutes: float) -> dict:
    """Record ETA observation for accuracy monitoring."""
    _collector.record_eta(predicted_minutes, actual_minutes)
    return {
        "recorded": True,
        "predicted": predicted_minutes,
        "actual": actual_minutes,
        "error": round(abs(predicted_minutes - actual_minutes), 2),
    }


@router.post(
    "/record/order",
    summary="Record an order outcome",
)
async def record_order(cancelled: bool = False) -> dict:
    """Record order for cancellation rate tracking."""
    _collector.record_order(cancelled)
    return {
        "total_orders": _collector.total_orders,
        "total_cancellations": _collector.total_cancellations,
        "rate": round(
            _collector.total_cancellations / max(_collector.total_orders, 1), 4
        ),
    }
