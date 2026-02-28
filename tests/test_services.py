"""
Integration tests for RPKOE services.

Tests service endpoints via FastAPI TestClient, exercising the full
pipeline from feature store through service logic to HTTP response.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.models import MerchantFeatures
from main import app, config, feature_store
from services import (
    dispatch_service,
    kitchen_state_service,
    monitoring_service,
    reliability_service,
)


@pytest.fixture(autouse=True)
def setup_services():
    """Initialize dependencies before each test."""
    kitchen_state_service.set_dependencies(feature_store, config)
    reliability_service.set_dependencies(feature_store, config)
    dispatch_service.set_dependencies(feature_store, config)
    monitoring_service.set_dependencies(config)

    # Seed feature store with a test merchant
    now = datetime.now()
    features = MerchantFeatures(
        merchant_id="test_m001",
        timestamp=now,
        active_orders=5,
        orders_last_5min=3,
        orders_last_15min=10,
        max_capacity=15,
        throughput_saturation=0.33,
        residual_drift_minutes=1.0,
        time_of_day_factor=0.7,
        marking_std=0.6,
        mark_on_arrival_rate=0.65,
        delay_entropy=0.8,
        rolling_pickup_delay_std=1.0,
        external_rush_factor=1.1,
        nearby_rider_count=5,
    )
    feature_store.set_merchant_features(features)

    yield

    feature_store.flush_all()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------


class TestSystem:
    def test_health(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_info(self, client: TestClient):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "congestion_weights" in data["config"]


# ---------------------------------------------------------------------------
# Kitchen State Service
# ---------------------------------------------------------------------------


class TestKitchenStateService:
    def test_estimate_state(self, client: TestClient):
        response = client.post("/kitchen/state?merchant_id=test_m001")
        assert response.status_code == 200
        data = response.json()
        assert "congestion_score" in data
        assert data["merchant_id"] == "test_m001"
        assert data["congestion_score"] >= 0

    def test_compute_kpt(self, client: TestClient):
        response = client.post("/kitchen/kpt?merchant_id=test_m001")
        assert response.status_code == 200
        data = response.json()
        assert data["mean_minutes"] > 0
        assert data["p50_minutes"] < data["p90_minutes"]

    def test_missing_merchant_404(self, client: TestClient):
        response = client.post("/kitchen/state?merchant_id=nonexistent")
        assert response.status_code == 404

    def test_calibration_record(self, client: TestClient):
        response = client.post(
            "/kitchen/calibration/record",
            params={
                "merchant_id": "test_m001",
                "predicted_p90": 15.0,
                "actual_kpt_minutes": 12.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "observed_coverage_p90" in data


# ---------------------------------------------------------------------------
# Reliability Service
# ---------------------------------------------------------------------------


class TestReliabilityService:
    def test_compute_mri(self, client: TestClient):
        response = client.post("/reliability/mri?merchant_id=test_m001")
        assert response.status_code == 200
        data = response.json()
        assert 0 < data["mri_score"] < 1

    def test_compute_kvi(self, client: TestClient):
        response = client.post("/reliability/kvi?merchant_id=test_m001")
        assert response.status_code == 200
        data = response.json()
        assert data["kvi_score"] >= 0

    def test_reliability_profile(self, client: TestClient):
        response = client.get("/reliability/profile/test_m001")
        assert response.status_code == 200
        data = response.json()
        assert "mri" in data
        assert "kvi" in data
        assert "interpretation" in data


# ---------------------------------------------------------------------------
# Dispatch Service
# ---------------------------------------------------------------------------


class TestDispatchService:
    def test_dispatch_decide(self, client: TestClient):
        response = client.post(
            "/dispatch/decide",
            json={
                "order_id": "test_order_1",
                "merchant_id": "test_m001",
                "expected_travel_time_minutes": 5.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        decision = data["decision"]

        assert decision["assign_delay_minutes"] >= 0
        assert decision["risk_level"] in ["LOW", "MODERATE", "HIGH_UNCERTAINTY"]
        assert len(decision["reason_codes"]) > 0
        assert "expected_cost" in data

    def test_dispatch_with_risk_override(self, client: TestClient):
        response = client.post(
            "/dispatch/decide",
            json={
                "order_id": "test_order_2",
                "merchant_id": "test_m001",
                "expected_travel_time_minutes": 5.0,
                "risk_lambda_override": 3.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["risk_lambda"] == 3.0

    def test_dispatch_with_low_supply(self, client: TestClient):
        response = client.post(
            "/dispatch/decide",
            json={
                "order_id": "test_order_3",
                "merchant_id": "test_m001",
                "expected_travel_time_minutes": 5.0,
                "nearby_rider_count": 1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Low supply should trigger supply adjustment
        assert data["decision"]["safety_buffer"]["supply_adjustment"] > 0

    def test_batch_dispatch(self, client: TestClient):
        response = client.post(
            "/dispatch/batch",
            json={
                "requests": [
                    {
                        "order_id": f"batch_{i}",
                        "merchant_id": "test_m001",
                        "expected_travel_time_minutes": 5.0,
                    }
                    for i in range(3)
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3


# ---------------------------------------------------------------------------
# Monitoring Service
# ---------------------------------------------------------------------------


class TestMonitoringService:
    def test_get_metrics(self, client: TestClient):
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "rider_wait_p50_minutes" in data
        assert "active_alerts" in data

    def test_record_rider_wait(self, client: TestClient):
        response = client.post("/monitoring/record/rider-wait?wait_minutes=3.5")
        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_record_eta(self, client: TestClient):
        response = client.post(
            "/monitoring/record/eta?predicted_minutes=10.0&actual_minutes=12.0"
        )
        assert response.status_code == 200
        assert response.json()["error"] == 2.0

    def test_record_order(self, client: TestClient):
        response = client.post("/monitoring/record/order?cancelled=false")
        assert response.status_code == 200
        assert response.json()["total_orders"] >= 1
