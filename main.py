"""
RPKOE — Real-Time Probabilistic Kitchen Orchestration Engine.

FastAPI application entry point. Mounts all service routers, initializes
shared dependencies (feature store, configuration), and provides a
simulation endpoint for end-to-end demonstration.

Production deployment architecture:
    - Each service would run as an independent Kubernetes deployment
    - Inter-service communication via gRPC or HTTP (with circuit breakers)
    - This monolith is the development/testing topology
    - Decomposition boundary: each router = one deployable service

Startup flow:
    1. Load configuration (from env vars in production)
    2. Initialize feature store (Redis cluster in production)
    3. Inject dependencies into all services
    4. Mount routers
    5. Ready to serve
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.optimization import (
    compute_congestion_with_memory,
    compute_kpt_distribution,
    compute_kvi_full,
    compute_mri_full,
    make_dispatch_decision,
)
from services import (
    dispatch_service,
    kitchen_state_service,
    monitoring_service,
    reliability_service,
)
from services.monitoring_service import get_collector
from stream.event_simulator import EventSimulator
from stream.feature_aggregator import FeatureAggregator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rpkoe")

# ---------------------------------------------------------------------------
# Shared state (initialized at startup)
# ---------------------------------------------------------------------------

config = RPKOEConfig()
feature_store = FeatureStore(config.partition)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize dependencies on startup."""
    logger.info("🚀 RPKOE starting up...")
    logger.info(f"   Partitions: {config.partition.num_partitions}")
    logger.info(f"   Merchants (simulation): {config.simulation.num_merchants}")
    logger.info(f"   Risk λ: {config.dispatch.risk_lambda}")

    # Inject dependencies into services
    kitchen_state_service.set_dependencies(feature_store, config)
    reliability_service.set_dependencies(feature_store, config)
    dispatch_service.set_dependencies(feature_store, config)
    monitoring_service.set_dependencies(config)

    logger.info("✅ All services initialized")
    yield
    logger.info("🛑 RPKOE shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RPKOE — Real-Time Probabilistic Kitchen Orchestration Engine",
    description=(
        "Production-grade system prototype for real-time kitchen state estimation "
        "and uncertainty-aware rider dispatch optimization. Designed for a food "
        "delivery marketplace operating at 300K+ merchants."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount service routers
app.include_router(kitchen_state_service.router)
app.include_router(reliability_service.router)
app.include_router(dispatch_service.router)
app.include_router(monitoring_service.router)


# ---------------------------------------------------------------------------
# Health & Info
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint for load balancer probes."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "feature_store_keys": feature_store.get_total_keys(),
        "partition_stats": feature_store.get_partition_stats(),
    }


@app.get("/info", tags=["System"])
async def info():
    """System configuration information."""
    return {
        "system": "RPKOE — Real-Time Probabilistic Kitchen Orchestration Engine",
        "config": {
            "congestion_weights": {
                "alpha": config.congestion.alpha,
                "beta": config.congestion.beta,
                "gamma": config.congestion.gamma,
                "delta": config.congestion.delta,
            },
            "decay_factor": config.congestion.decay_factor,
            "risk_lambda": config.dispatch.risk_lambda,
            "safety_buffer": {
                "a": config.safety.a,
                "b": config.safety.b,
                "c": config.safety.c,
            },
            "mri_weights": {
                "w1": config.mri.w1,
                "w2": config.mri.w2,
                "w3": config.mri.w3,
            },
            "partitions": config.partition.num_partitions,
            "windows": {
                "short": f"{config.windows.short_window_minutes}min",
                "long": f"{config.windows.long_window_minutes}min",
            },
        },
    }


# ---------------------------------------------------------------------------
# End-to-End Simulation
# ---------------------------------------------------------------------------


@app.post("/simulate", tags=["Simulation"])
async def run_simulation(
    duration_minutes: int = 30,
    num_merchants: int | None = None,
) -> dict:
    """Run a full end-to-end simulation.

    This endpoint exercises the entire RPKOE pipeline:
        1. Generates synthetic event stream
        2. Processes events through feature aggregator (with idempotency)
        3. Computes features for all merchants
        4. Runs dispatch decisions
        5. Records metrics
        6. Returns summary with example decisions and system metrics

    In production, this pipeline runs continuously via Kafka consumers.
    This endpoint simulates the same flow for demonstration.
    """
    logger.info(f"🎬 Starting simulation: {duration_minutes}min")

    # Override merchant count if specified
    sim_config = config
    if num_merchants is not None:
        sim_config = config.model_copy(deep=True)
        sim_config.simulation.num_merchants = num_merchants

    # Initialize simulator and aggregator
    simulator = EventSimulator(sim_config, start_hour=12.0)
    aggregator = FeatureAggregator(feature_store, sim_config)
    aggregator.initialize_merchants(simulator.get_merchant_profiles())
    collector = get_collector()

    # Process event stream
    event_count = 0
    async for event in simulator.stream(duration_minutes=duration_minutes):
        aggregator.process_event(event)
        event_count += 1

        # Periodic feature flush (every ~100 events)
        if event_count % 100 == 0:
            aggregator.flush_features(current_time=event.timestamp)

    # Final feature flush
    aggregator.flush_features()
    logger.info(f"📊 Processed {event_count} events")

    # Run dispatch decisions for all merchants with features
    dispatch_results = []
    merchant_ids = feature_store.get_all_merchant_ids()

    for mid in merchant_ids[:20]:  # Sample 20 merchants for demo
        features = feature_store.get_merchant_features(mid)
        if features is None:
            continue

        prev_congestion = feature_store.get_previous_congestion(mid)
        kitchen_state = compute_congestion_with_memory(
            features, prev_congestion, sim_config
        )
        feature_store.set_previous_congestion(mid, kitchen_state.congestion_score)

        kpt_dist = compute_kpt_distribution(
            kitchen_state.congestion_score, mid, features.timestamp, sim_config
        )

        sample_count = int(feature_store.get(mid, "for_sample_count", default=10))
        mri = compute_mri_full(features, sample_count, sim_config)
        kvi = compute_kvi_full(features)

        # Simulated travel time
        travel_time = 3.0 + features.throughput_saturation * 2.0

        decision = make_dispatch_decision(
            order_id=str(uuid.uuid4())[:8],
            merchant_id=mid,
            features=features,
            kitchen_state=kitchen_state,
            kpt_dist=kpt_dist,
            mri=mri,
            kvi=kvi,
            expected_travel_time=travel_time,
            config=sim_config,
        )

        dispatch_results.append(decision.model_dump())

        # Record monitoring observations
        simulated_wait = max(0, decision.predicted_kpt_minutes - travel_time - decision.assign_delay_minutes)
        collector.record_rider_wait(simulated_wait)
        collector.record_eta(decision.predicted_kpt_minutes, decision.predicted_kpt_minutes * (1 + features.residual_drift_minutes / 10))
        collector.record_order(cancelled=False)
        collector.record_dispatch()
        collector.record_mri(mid, mri.mri_score)
        collector.record_calibration(mid, kpt_dist.p90_minutes, kpt_dist.mean_minutes * 1.1)

    # Get aggregation stats
    agg_stats = aggregator.get_aggregation_stats()

    logger.info(f"✅ Simulation complete: {len(dispatch_results)} dispatch decisions")

    return {
        "simulation": {
            "duration_minutes": duration_minutes,
            "total_events": event_count,
            "merchants_with_features": len(merchant_ids),
            "dispatch_decisions": len(dispatch_results),
        },
        "aggregation_stats": agg_stats,
        "feature_store": {
            "total_keys": feature_store.get_total_keys(),
            "partition_distribution": feature_store.get_partition_stats(),
        },
        "sample_dispatch_decisions": dispatch_results[:5],
        "partition_note": (
            f"Events are partitioned across {sim_config.partition.num_partitions} "
            f"partitions by hash(merchant_id). Each partition owns a disjoint set "
            f"of merchants, enabling independent horizontal scaling."
        ),
    }
