"""
Unit tests for the core optimization module.

Tests all pure mathematical functions independently with known inputs
and expected outputs. No I/O, no services, no feature store.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.config import RPKOEConfig
from core.models import (
    KitchenState,
    KPTDistribution,
    MerchantFeatures,
    MerchantReliability,
    RiskLevel,
)
from core.optimization import (
    _sigmoid,
    compute_assign_delay,
    compute_congestion_with_memory,
    compute_dispatch_cost,
    compute_expected_cost,
    compute_instantaneous_congestion,
    compute_kpt_distribution,
    compute_kvi,
    compute_kvi_full,
    compute_mri,
    compute_mri_full,
    compute_safety_buffer,
    make_dispatch_decision,
    monte_carlo_optimize_lambda,
)


@pytest.fixture
def config() -> RPKOEConfig:
    return RPKOEConfig()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 2, 28, 13, 0, 0)


@pytest.fixture
def base_features(now: datetime) -> MerchantFeatures:
    """Baseline merchant features for testing."""
    return MerchantFeatures(
        merchant_id="test_merchant",
        timestamp=now,
        active_orders=5,
        orders_last_5min=3,
        orders_last_15min=10,
        max_capacity=15,
        throughput_saturation=0.33,
        residual_drift_minutes=1.0,
        time_of_day_factor=0.8,
        marking_std=0.5,
        mark_on_arrival_rate=0.6,
        delay_entropy=0.7,
        rolling_pickup_delay_std=1.2,
        external_rush_factor=1.1,
        nearby_rider_count=5,
    )


# ---------------------------------------------------------------------------
# Congestion Tests
# ---------------------------------------------------------------------------


class TestCongestion:
    """Tests for kitchen congestion estimation."""

    def test_instantaneous_congestion_baseline(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """Congestion score should be positive for non-zero inputs."""
        score = compute_instantaneous_congestion(base_features, config)
        assert score > 0
        assert score < 2.0  # Reasonable upper bound

    def test_congestion_increases_with_load(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """Higher active orders → higher congestion."""
        low_load = base_features.model_copy(update={"active_orders": 2, "throughput_saturation": 0.13})
        high_load = base_features.model_copy(update={"active_orders": 14, "throughput_saturation": 0.93})

        score_low = compute_instantaneous_congestion(low_load, config)
        score_high = compute_instantaneous_congestion(high_load, config)

        assert score_high > score_low

    def test_stateful_congestion_cold_start(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """First estimation (no previous state) uses instantaneous score."""
        state = compute_congestion_with_memory(base_features, None, config)
        instantaneous = compute_instantaneous_congestion(base_features, config)

        assert isinstance(state, KitchenState)
        assert state.congestion_score == pytest.approx(instantaneous, abs=0.2)

    def test_stateful_congestion_with_memory(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """With previous state, result should be a blend."""
        prev = 0.8
        state = compute_congestion_with_memory(base_features, prev, config)

        # Should be between previous and instantaneous
        instantaneous = compute_instantaneous_congestion(base_features, config)
        assert state.congestion_score >= min(prev, instantaneous) - 0.2
        assert state.congestion_score <= max(prev, instantaneous) + 0.3

    def test_spike_detection(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """Spike detected when residual drift exceeds threshold."""
        spiked = base_features.model_copy(update={"residual_drift_minutes": 5.0})
        state = compute_congestion_with_memory(spiked, None, config)
        assert state.spike_detected is True

    def test_no_spike_normal_drift(
        self, base_features: MerchantFeatures, config: RPKOEConfig
    ):
        """No spike for normal residual drift."""
        normal = base_features.model_copy(update={"residual_drift_minutes": 0.5})
        state = compute_congestion_with_memory(normal, None, config)
        assert state.spike_detected is False


# ---------------------------------------------------------------------------
# KPT Distribution Tests
# ---------------------------------------------------------------------------


class TestKPTDistribution:
    """Tests for the LogNormal KPT distribution."""

    def test_kpt_produces_valid_distribution(self, config: RPKOEConfig, now: datetime):
        """KPT distribution should have valid percentiles."""
        kpt = compute_kpt_distribution(0.5, "test", now, config)

        assert isinstance(kpt, KPTDistribution)
        assert kpt.mean_minutes > 0
        assert kpt.std_minutes > 0
        assert kpt.p50_minutes < kpt.p75_minutes < kpt.p90_minutes < kpt.p95_minutes

    def test_kpt_increases_with_congestion(self, config: RPKOEConfig, now: datetime):
        """Higher congestion should increase KPT mean and spread."""
        low = compute_kpt_distribution(0.1, "test", now, config)
        high = compute_kpt_distribution(0.9, "test", now, config)

        assert high.mean_minutes > low.mean_minutes
        assert high.std_minutes > low.std_minutes

    def test_kpt_p90_greater_than_mean(self, config: RPKOEConfig, now: datetime):
        """LogNormal is right-skewed: P90 > mean > P50."""
        kpt = compute_kpt_distribution(0.5, "test", now, config)
        assert kpt.p90_minutes > kpt.mean_minutes
        assert kpt.mean_minutes > kpt.p50_minutes


# ---------------------------------------------------------------------------
# MRI Tests
# ---------------------------------------------------------------------------


class TestMRI:
    """Tests for Merchant Reliability Index."""

    def test_mri_in_range(self, config: RPKOEConfig):
        """MRI should always be in (0, 1)."""
        mri = compute_mri(0.5, 0.6, 0.7, config)
        assert 0 < mri < 1

    def test_mri_higher_for_reliable_merchant(self, config: RPKOEConfig):
        """Reliable merchant (low std, high mark-on-arrival) → higher MRI."""
        reliable = compute_mri(0.2, 0.9, 0.3, config)
        unreliable = compute_mri(2.0, 0.2, 2.0, config)
        assert reliable > unreliable

    def test_sigmoid_properties(self):
        """Sigmoid should be monotonic and bounded."""
        assert _sigmoid(0) == pytest.approx(0.5)
        assert _sigmoid(10) > 0.99
        assert _sigmoid(-10) < 0.01


# ---------------------------------------------------------------------------
# KVI Tests
# ---------------------------------------------------------------------------


class TestKVI:
    """Tests for Kitchen Volatility Index."""

    def test_kvi_scales_with_rush(self):
        """KVI should scale linearly with rush factor."""
        base = compute_kvi(1.0, 1.0)
        rushed = compute_kvi(1.0, 2.0)
        assert rushed == pytest.approx(2 * base)

    def test_kvi_non_negative(self):
        """KVI should never be negative."""
        assert compute_kvi(0.0, 1.0) >= 0
        assert compute_kvi(-1.0, 1.0) >= 0  # Clamped


# ---------------------------------------------------------------------------
# Safety Buffer Tests
# ---------------------------------------------------------------------------


class TestSafetyBuffer:
    """Tests for adaptive dispatch safety buffer."""

    def test_buffer_increases_with_uncertainty(self, config: RPKOEConfig):
        """Higher KPT std → larger safety buffer."""
        low = compute_safety_buffer(0.5, 0.5, 0.8, 5, config)
        high = compute_safety_buffer(3.0, 0.5, 0.8, 5, config)
        assert high.total > low.total

    def test_buffer_increases_with_unreliability(self, config: RPKOEConfig):
        """Lower MRI → larger safety buffer."""
        reliable = compute_safety_buffer(1.0, 0.5, 0.9, 5, config)
        unreliable = compute_safety_buffer(1.0, 0.5, 0.3, 5, config)
        assert unreliable.total > reliable.total

    def test_supply_adjustment_reduces_buffer(self, config: RPKOEConfig):
        """Low rider supply should reduce the safety buffer."""
        normal_supply = compute_safety_buffer(1.0, 1.0, 0.5, 10, config)
        low_supply = compute_safety_buffer(1.0, 1.0, 0.5, 1, config)

        assert low_supply.supply_adjustment > 0
        assert low_supply.total < normal_supply.total

    def test_buffer_component_breakdown(self, config: RPKOEConfig):
        """Buffer breakdown components should sum to total (without supply adj)."""
        buf = compute_safety_buffer(1.0, 1.0, 0.5, 10, config)
        component_sum = buf.kpt_std_component + buf.kvi_component + buf.mri_component
        assert buf.total == pytest.approx(component_sum, abs=0.01)


# ---------------------------------------------------------------------------
# Dispatch Decision Tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Tests for the full dispatch decision pipeline."""

    def test_assign_delay_non_negative(self):
        """Assignment delay should never be negative."""
        assert compute_assign_delay(10.0, 5.0, 2.0) >= 0
        assert compute_assign_delay(5.0, 10.0, 2.0) == 0  # Travel > prep

    def test_dispatch_cost_no_late(self):
        """Cost without late penalty is just rider wait."""
        cost = compute_dispatch_cost(3.0, False, 1.5)
        assert cost == 3.0

    def test_dispatch_cost_with_late(self):
        """Cost with late penalty includes λ * 1.0."""
        cost = compute_dispatch_cost(3.0, True, 1.5)
        assert cost == 3.0 + 1.5

    def test_full_dispatch_decision(
        self, base_features: MerchantFeatures, config: RPKOEConfig, now: datetime
    ):
        """End-to-end dispatch decision should produce valid output."""
        state = compute_congestion_with_memory(base_features, None, config)
        kpt = compute_kpt_distribution(state.congestion_score, "test", now, config)
        mri = compute_mri_full(base_features, 20, config)
        kvi = compute_kvi_full(base_features)

        decision = make_dispatch_decision(
            order_id="test_order",
            merchant_id="test_merchant",
            features=base_features,
            kitchen_state=state,
            kpt_dist=kpt,
            mri=mri,
            kvi=kvi,
            expected_travel_time=5.0,
            config=config,
        )

        assert decision.order_id == "test_order"
        assert decision.assign_delay_minutes >= 0
        assert decision.risk_level in RiskLevel
        assert len(decision.reason_codes) > 0
        assert decision.safety_buffer.total >= 0

    def test_high_uncertainty_risk_level(
        self, config: RPKOEConfig, now: datetime
    ):
        """Very uncertain merchant should produce HIGH_UNCERTAINTY risk."""
        features = MerchantFeatures(
            merchant_id="unreliable",
            timestamp=now,
            active_orders=14,
            max_capacity=15,
            throughput_saturation=0.93,
            residual_drift_minutes=4.0,
            time_of_day_factor=0.9,
            marking_std=3.0,
            mark_on_arrival_rate=0.1,
            delay_entropy=2.5,
            rolling_pickup_delay_std=4.0,
            external_rush_factor=1.5,
            nearby_rider_count=2,
        )

        state = compute_congestion_with_memory(features, None, config)
        kpt = compute_kpt_distribution(state.congestion_score, "unreliable", now, config)
        mri = compute_mri_full(features, 5, config)
        kvi = compute_kvi_full(features)

        decision = make_dispatch_decision(
            order_id="risky_order",
            merchant_id="unreliable",
            features=features,
            kitchen_state=state,
            kpt_dist=kpt,
            mri=mri,
            kvi=kvi,
            expected_travel_time=5.0,
            config=config,
        )

        assert decision.risk_level == RiskLevel.HIGH_UNCERTAINTY


# ---------------------------------------------------------------------------
# Monte Carlo Tests
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """Tests for offline parameter optimization."""

    def test_monte_carlo_returns_valid_result(self):
        """Monte Carlo should return optimal λ and per-lambda costs."""
        result = monte_carlo_optimize_lambda(
            kpt_mu=2.0,
            kpt_sigma=0.3,
            travel_time=5.0,
            safety_buffer=2.0,
            n_samples=1000,
        )

        assert "optimal_lambda" in result
        assert "results_by_lambda" in result
        assert result["n_samples"] == 1000

        for lam, data in result["results_by_lambda"].items():
            assert "expected_cost" in data
            assert "late_probability" in data
            assert 0 <= data["late_probability"] <= 1
