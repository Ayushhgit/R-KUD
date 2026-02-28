"""
Centralized system configuration for RPKOE.

All tunable parameters are defined here using Pydantic BaseSettings, enabling
environment-variable overrides in production while providing sensible defaults
for development. Parameters are grouped by subsystem concern.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class CongestionCoefficients(BaseSettings):
    """Weights for the hidden kitchen congestion state estimator.

    congestion_t = α * active_orders_norm
                 + β * residual_drift
                 + γ * time_of_day_factor
                 + δ * throughput_saturation
    """

    alpha: float = Field(0.35, description="Weight for normalized active order count")
    beta: float = Field(0.25, description="Weight for residual drift (actual - estimated)")
    gamma: float = Field(0.20, description="Weight for time-of-day demand factor")
    delta: float = Field(0.20, description="Weight for throughput saturation ratio")

    # Stateful estimation: exponential decay memory
    decay_factor: float = Field(
        0.7,
        description="Exponential decay for previous congestion state. "
        "congestion_t = decay * congestion_(t-1) + (1 - decay) * instantaneous",
    )
    spike_penalty: float = Field(
        0.15,
        description="Additive penalty when residual drift exceeds spike threshold",
    )
    spike_threshold: float = Field(
        2.0,
        description="Residual drift (minutes) above which spike penalty triggers",
    )


class KPTDistributionConfig(BaseSettings):
    """Parameters for the LogNormal KPT distribution model.

    KPT ~ LogNormal(μ(congestion_t), σ(congestion_t))
    μ and σ are affine functions of the congestion score.
    """

    mu_base: float = Field(2.0, description="Base log-mean (≈ exp(2) ≈ 7.4 min)")
    mu_congestion_scale: float = Field(0.5, description="μ increase per unit congestion")
    sigma_base: float = Field(0.3, description="Base log-std")
    sigma_congestion_scale: float = Field(0.2, description="σ increase per unit congestion")

    # Calibration guardrails
    calibration_window_size: int = Field(
        100, description="Recent observations used for CRPS / interval calibration"
    )
    expected_coverage_p90: float = Field(
        0.90, description="Expected fraction of actuals inside P90 interval"
    )
    calibration_alert_threshold: float = Field(
        0.05, description="Allowable deviation from expected coverage before alert"
    )


class SafetyBufferConfig(BaseSettings):
    """Weights for the adaptive dispatch safety buffer.

    safety_buffer = a * kpt_std + b * KVI + c * (1 - MRI)
    """

    a: float = Field(1.5, description="Sensitivity to KPT uncertainty (std)")
    b: float = Field(1.0, description="Sensitivity to Kitchen Volatility Index")
    c: float = Field(2.0, description="Sensitivity to merchant unreliability (1 - MRI)")


class DispatchConfig(BaseSettings):
    """Dispatch optimization parameters.

    Objective: Minimize E[rider_wait_cost] + λ * E[late_penalty]
    """

    risk_lambda: float = Field(
        1.5,
        description="Risk aversion parameter λ. Higher = more conservative dispatch",
    )

    # Risk level thresholds (on safety_buffer)
    high_uncertainty_threshold: float = Field(
        5.0, description="Safety buffer above which risk_level = HIGH_UNCERTAINTY"
    )
    moderate_uncertainty_threshold: float = Field(
        2.5, description="Safety buffer above which risk_level = MODERATE"
    )

    # Supply-awareness
    low_supply_buffer_reduction: float = Field(
        0.3,
        description="Fraction by which safety buffer is reduced when rider supply is low",
    )
    low_supply_threshold: int = Field(
        3, description="Nearby rider count below which supply is considered low"
    )


class MRIConfig(BaseSettings):
    """Configuration for Merchant Reliability Index computation.

    Composite MRI uses a sigmoid over weighted features:
    MRI = sigmoid(w1 * inv_std + w2 * mark_on_arrival_rate + w3 * inv_entropy)
    """

    w1: float = Field(0.4, description="Weight for inverse marking std")
    w2: float = Field(0.35, description="Weight for mark-on-arrival probability")
    w3: float = Field(0.25, description="Weight for inverse delay entropy")


class FeatureWindowConfig(BaseSettings):
    """Sliding window configuration for feature aggregation."""

    short_window_minutes: int = Field(5, description="Short sliding window (minutes)")
    long_window_minutes: int = Field(15, description="Long sliding window (minutes)")
    max_buffer_size: int = Field(
        500, description="Max events per merchant in memory buffer"
    )


class MonitoringConfig(BaseSettings):
    """Thresholds for monitoring guardrails."""

    rider_wait_p90_alert_minutes: float = Field(
        8.0, description="P90 rider wait threshold triggering alert"
    )
    eta_p90_baseline_minutes: float = Field(
        25.0, description="Baseline ETA P90; alert if exceeded by > 2 min"
    )
    eta_p90_alert_buffer_minutes: float = Field(
        2.0, description="Buffer above ETA baseline before alert fires"
    )
    cancellation_rate_alert_threshold: float = Field(
        0.05, description="Cancellation rate above which alert fires"
    )
    cancellation_spike_pct: float = Field(
        0.20, description="20% spike in cancellation rate triggers rollback alert"
    )
    congestion_drift_alert_threshold: float = Field(
        0.3, description="Sustained drift in congestion predictions vs actuals"
    )
    mri_degradation_threshold: float = Field(
        0.15, description="MRI drop threshold triggering alert"
    )


class SimulationConfig(BaseSettings):
    """Event simulation parameters."""

    num_merchants: int = Field(50, description="Number of merchants in simulation")
    merchant_max_capacity: int = Field(15, description="Max concurrent orders per merchant")
    simulation_duration_minutes: int = Field(60, description="Simulation duration")
    base_order_rate_per_minute: float = Field(
        0.5, description="Base order arrival rate per merchant per minute"
    )
    rider_speed_kmh: float = Field(25.0, description="Average rider speed in km/h")
    avg_pickup_distance_km: float = Field(2.0, description="Average pickup distance")


class PartitionConfig(BaseSettings):
    """Partitioning configuration for horizontal scaling.

    In production, Kafka topics and feature store are partitioned by:
        partition_id = hash(merchant_id) % num_partitions

    Each partition owns a disjoint subset of merchants, enabling:
    - Independent scaling of feature aggregation workers
    - Localized state management (rolling windows, MRI history)
    - Linear throughput scaling with partition count
    """

    num_partitions: int = Field(
        16,
        description="Number of logical partitions. "
        "In production, this maps to Kafka partitions and Redis cluster slots.",
    )

    def get_partition(self, merchant_id: str) -> int:
        """Deterministic partition assignment for a merchant.

        Ensures all events for the same merchant land on the same partition,
        preserving ordering guarantees within the merchant scope.
        """
        return hash(merchant_id) % self.num_partitions


class RPKOEConfig(BaseSettings):
    """Root configuration aggregating all subsystem configs."""

    congestion: CongestionCoefficients = CongestionCoefficients()
    kpt: KPTDistributionConfig = KPTDistributionConfig()
    safety: SafetyBufferConfig = SafetyBufferConfig()
    dispatch: DispatchConfig = DispatchConfig()
    mri: MRIConfig = MRIConfig()
    windows: FeatureWindowConfig = FeatureWindowConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    simulation: SimulationConfig = SimulationConfig()
    partition: PartitionConfig = PartitionConfig()

    model_config = {"env_prefix": "RPKOE_"}
