# RPKOE — Real-Time Probabilistic Kitchen Orchestration Engine

> Production-grade system prototype for real-time kitchen state estimation and uncertainty-aware rider dispatch optimization in a food delivery marketplace.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Code Structure](#code-structure)
- [Scalability Strategy](#scalability-strategy)
- [Monitoring & Guardrails](#monitoring--guardrails)
- [Pilot & Rollout Plan](#pilot--rollout-plan)
- [Running the System](#running-the-system)

---

## High-Level Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVENT STREAM LAYER (Kafka)                        │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐                  │
│   │ orders.events│  │riders.telem │  │ merchants.for    │                  │
│   │  (partitioned│  │  (partitioned│  │  (partitioned    │                  │
│   │  by merchant)│  │  by merchant)│  │  by merchant)    │                  │
│   └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘                  │
└──────────┼────────────────┼──────────────────┼─────────────────────────────┘
           │                │                  │
           ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STREAMING FEATURE LAYER                                  │
│                                                                             │
│   ┌──────────────────────────────────────────────┐                         │
│   │         Feature Aggregator (N instances)      │                         │
│   │                                                │                         │
│   │  • Idempotent event processing (event_id)     │                         │
│   │  • Event ordering with late-event handling     │                         │
│   │  • 5-min / 15-min sliding window aggregation   │                         │
│   │  • Rolling residual drift computation          │                         │
│   │  • Shannon entropy for delay characterization  │                         │
│   │  • Partition-owned: hash(merchant_id) % N      │                         │
│   └──────────────────┬───────────────────────────┘                         │
│                      │                                                      │
│                      ▼                                                      │
│   ┌──────────────────────────────────────────────┐                         │
│   │       Feature Store (Redis Cluster)           │                         │
│   │                                                │                         │
│   │  • MerchantFeatures per merchant              │                         │
│   │  • Congestion state memory (decay)            │                         │
│   │  • TTL-based expiry                            │                         │
│   │  • Per-partition locking                       │                         │
│   └──────────────────┬───────────────────────────┘                         │
└──────────────────────┼─────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌───────────┐ ┌──────────────┐
│Kitchen State │ │Reliability│ │  Dispatch     │
│  Service     │ │  Service  │ │  Control      │
│              │ │           │ │  Engine       │
│• Congestion  │ │• MRI      │ │              │
│  estimation  │ │  (sigmoid)│ │• Safety      │
│  (stateful)  │ │• KVI      │ │  buffer      │
│• KPT distrib │ │  (scaled) │ │• Risk class. │
│  (LogNormal) │ │• Profile  │ │• Supply-aware│
│• Calibration │ │           │ │• Reason codes│
│  monitoring  │ │           │ │• Monte Carlo │
└──────────────┘ └───────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
         ┌──────────────────────────┐
         │ Monitoring & Guardrails  │
         │                          │
         │ • Rider wait P50/P90     │
         │ • ETA accuracy P90       │
         │ • Calibration drift      │
         │ • Cancellation spikes    │
         │ • MRI degradation        │
         │ • Congestion drift       │
         └──────────────────────────┘
```

### Data Flow

1. **Ingest** — Order, rider, and FOR events arrive via partitioned Kafka topics
2. **Aggregate** — Feature aggregator computes sliding-window features per merchant with idempotency and ordering guarantees
3. **Estimate** — Kitchen State Service computes stateful congestion scores with exponential decay memory and spike detection
4. **Distribute** — KPT Distribution Service produces LogNormal probabilistic prep time estimates
5. **Score** — Reliability Service computes composite MRI and KVI per merchant
6. **Dispatch** — Dispatch Engine uses all signals to compute supply-aware, uncertainty-aware rider assignment delays with full explainability
7. **Monitor** — Monitoring Service tracks 6 guardrail metrics and fires alerts

### Latency Constraints

| Service | P99 Target | Justification |
|---------|-----------|---------------|
| Feature Aggregator | < 10ms per event | High-throughput streaming path |
| Kitchen State | < 20ms | Called per dispatch decision |
| Reliability | < 15ms | Cacheable (5-min TTL) |
| Dispatch Engine | < 50ms | Critical path for every order |
| Monitoring | < 100ms | Async, non-blocking |

### Storage Strategy

| Data | Store | TTL | Notes |
|------|-------|-----|-------|
| Merchant Features | Redis Cluster | 60s | Hot path, per-partition sharded |
| Congestion Memory | Redis (sorted set) | 30min | Per-merchant decay state |
| MRI History | Redis + TimescaleDB | 24h / ∞ | Hot for serving, cold for analysis |
| Calibration Data | TimescaleDB | 7d | Rolling CRPS computation |
| Event Stream | Kafka (7d retention) | 7d | Replay for reprocessing |

### Failure Handling

- **Circuit Breakers**: Each inter-service call uses circuit breakers with fallback to cached values
- **Backpressure**: Feature aggregator buffers under load; signals Kafka to pause at high-water mark
- **Cold Start**: New merchants use population-level priors until 20+ observations accumulate
- **Stale Features**: TTL-based expiry ensures stale features are never served; fallback to conservative defaults
- **Duplicate Events**: Idempotent processing via event_id tracking in feature store
- **Late Events**: Processed but inserted into correct window position (not appended)

---

## Mathematical Formulation

### 1. Hidden Kitchen Congestion State

The kitchen congestion is modeled as a hidden dynamic variable with exponential decay memory:

```
congestion_inst = α · (active_orders / max_capacity)
               + β · normalize(residual_drift)
               + γ · time_of_day_factor
               + δ · throughput_saturation

congestion_t = λ_decay · congestion_(t-1) + (1 - λ_decay) · congestion_inst
```

**Spike Detection:**
```
if |residual_drift| > spike_threshold:
    congestion_t += spike_penalty
```

Default coefficients: α=0.35, β=0.25, γ=0.20, δ=0.20, λ_decay=0.7

### 2. KPT Distribution (LogNormal)

```
KPT ~ LogNormal(μ(c_t), σ(c_t))

where:
    μ = μ_base + μ_scale · congestion_t
    σ = σ_base + σ_scale · congestion_t

E[KPT] = exp(μ + σ²/2)
Var[KPT] = (exp(σ²) - 1) · exp(2μ + σ²)
```

The LogNormal is chosen because preparation times are:
- Strictly positive
- Right-skewed (occasional long waits)
- Mode < Mean (most orders faster than average)

### 3. Merchant Reliability Index (MRI)

```
MRI = σ(w₁ · 1/(1 + marking_std)
      + w₂ · mark_on_arrival_rate
      + w₃ · 1/(1 + delay_entropy))
```

Where σ is the sigmoid function. Captures:
- **Consistency**: Low marking_std → reliable FOR signaling
- **Accuracy**: High mark-on-arrival rate → food ready when marked
- **Predictability**: Low delay entropy → predictable delay distribution

### 4. Kitchen Volatility Index (KVI)

```
KVI = rolling_std(pickup_delay) × external_rush_factor
```

### 5. Dispatch Optimization

**Objective:**
```
Minimize: E[rider_wait_cost] + λ · E[late_penalty]
```

**Safety Buffer (supply-aware):**
```
buffer_raw = a · σ_KPT + b · KVI + c · (1 - MRI)

if nearby_riders < low_supply_threshold:
    buffer = buffer_raw × (1 - supply_reduction_factor)
else:
    buffer = buffer_raw
```

**Assignment Delay:**
```
assign_delay = max(0, E[KPT] - expected_travel_time - safety_buffer)
```

**How uncertainty influences dispatch:**
- High σ_KPT → larger safety buffer → earlier dispatch (rider waits more but won't be late)
- Low MRI → larger safety buffer → more conservative
- Low rider supply → reduced buffer → accept more risk to avoid stranding orders
- High λ → more penalty for late arrival → more conservative dispatch

### 6. Distribution Calibration

```
calibration_error = observed_coverage_p90 - expected_coverage_p90

if |calibration_error| > threshold:
    trigger DRIFT_ALERT
    action: recalibrate σ parameters
```

---

## Code Structure

```
rpkoe/
├── pyproject.toml              # Project definition (uv)
├── main.py                     # FastAPI app, routing, simulation
│
├── core/
│   ├── config.py               # All tunable parameters (Pydantic Settings)
│   ├── models.py               # 20+ Pydantic data contracts
│   ├── optimization.py         # Pure math (congestion, KPT, MRI, KVI, dispatch)
│   └── feature_store.py        # Redis-simulated store (TTL, partitions, idempotency)
│
├── stream/
│   ├── event_simulator.py      # Kafka-like event generator (TOD, noise, duplicates)
│   └── feature_aggregator.py   # Sliding window aggregation (idempotent, ordered)
│
├── services/
│   ├── kitchen_state_service.py    # Stateful congestion + KPT + calibration
│   ├── reliability_service.py      # Composite MRI + KVI
│   ├── dispatch_service.py         # Supply-aware dispatch + Monte Carlo
│   └── monitoring_service.py       # 6 guardrails + metrics + alerts
│
└── tests/
    ├── test_optimization.py    # 24 unit tests (pure math)
    └── test_services.py        # 17 integration tests (endpoints)
```

Each service follows the dependency injection pattern (set at startup, no global coupling) and is stateless (all state in feature store).

---

## Scalability Strategy

### Handling 300K Merchants

The system scales horizontally via **partitioned event processing** keyed by `merchant_id`:

```
partition_id = hash(merchant_id) % N
```

Each partition owns:
- Merchant feature state (sliding windows)
- Rolling MRI history
- KVI history
- Congestion decay memory

### Horizontal Scaling

| Component | Scaling Strategy |
|-----------|-----------------|
| Kafka Topics | Partition by merchant_id; add partitions as merchant count grows |
| Feature Aggregator | 1 instance per partition range; Kubernetes HPA on CPU |
| Feature Store | Redis Cluster with consistent hashing; auto-resharding |
| Kitchen State Service | Stateless; scale behind L7 load balancer |
| Dispatch Engine | Stateless; scale to match order throughput |
| Monitoring | Single instance with async metric collection |

### Peak Handling (Lunch/Dinner)

- **Pre-scaling**: Kubernetes CronJob scales up pods 15 min before known peaks
- **Burst handling**: Feature aggregator buffers under backpressure
- **Cache warming**: MRI/KVI cached with 5-min TTL (changes slowly)
- **Batch dispatch**: Process dispatch decisions in 15-second windows during peaks

### Cold Start for New Restaurants

- Use population-level priors: μ_KPT = median across similar cuisine/location
- MRI starts at 0.5 (neutral) until 20+ FOR observations
- Gradually shift from priors to observed statistics using Bayesian updating
- Flag new merchants with `COLD_START` reason code in dispatch decisions

---

## Monitoring & Guardrails

### Metrics Dashboard

| Metric | Target | Alert Trigger | Action |
|--------|--------|---------------|--------|
| Rider Wait P90 | < 8 min | > 8 min | Review safety buffer params |
| ETA P90 | < 27 min | > baseline + 2 min | Investigate congestion model |
| Cancellation Rate | < 5% | > 5% | Consider param rollback |
| Cancellation Spike | — | > 20% increase | AUTO-ROLLBACK dispatch |
| Calibration Coverage | ~90% | |error| > 5% | Trigger recalibration |
| Congestion Drift | < 0.3 | > 0.3 | Review model coefficients |
| MRI Degradation | — | > 15% drop | Flag merchants for review |

### Example Prometheus Metrics

```
# TYPE rpkoe_rider_wait_seconds histogram
rpkoe_rider_wait_seconds_bucket{le="60"} 1234
rpkoe_rider_wait_seconds_bucket{le="120"} 2345

# TYPE rpkoe_dispatch_risk_level counter
rpkoe_dispatch_risk_level{level="LOW"} 8742
rpkoe_dispatch_risk_level{level="MODERATE"} 3219
rpkoe_dispatch_risk_level{level="HIGH_UNCERTAINTY"} 891

# TYPE rpkoe_calibration_error gauge
rpkoe_calibration_error{merchant_cluster="north_indian"} 0.03
rpkoe_calibration_error{merchant_cluster="fast_food"} -0.02

# TYPE rpkoe_mri_score histogram
rpkoe_mri_score_bucket{le="0.3"} 45
rpkoe_mri_score_bucket{le="0.5"} 123
rpkoe_mri_score_bucket{le="0.7"} 890
rpkoe_mri_score_bucket{le="1.0"} 1200
```

---

## Pilot & Rollout Plan

### A/B Testing Design

| Dimension | Control | Treatment |
|-----------|---------|-----------|
| Dispatch Logic | Fixed safety buffer (3 min) | Adaptive (uncertainty-aware) |
| Splitting | By city zone (geographic) | Same zones, alternate days |
| Duration | 14 days | 14 days |
| Sample Size | ~50K orders/day | ~50K orders/day |
| Primary Metric | Rider wait time P90 | Rider wait time P90 |
| Secondary Metrics | Late arrival %, cancellation rate, rider utilization |

### Guardrail Metrics

- **Hard guardrails** (auto-rollback if breached):
  - Cancellation rate increase > 20%
  - Late arrival rate increase > 15%
  - Rider wait P90 increase > 2 min vs control

- **Soft guardrails** (alert for manual review):
  - ETA accuracy degradation > 10%
  - MRI computation failures > 1%
  - Feature store latency P99 > 50ms

### Progressive Rollout

```
Phase 1 (Week 1-2):  Shadow mode — compute decisions, log, don't act
Phase 2 (Week 3-4):  5% traffic in 2 cities with most reliable merchants
Phase 3 (Week 5-6):  20% traffic, expand to 5 cities
Phase 4 (Week 7-8):  50% traffic with A/B measurement
Phase 5 (Week 9+):   100% with automated guardrails
```

### Fallback Mechanism

```
if guardrail_breached:
    1. Revert to fixed safety_buffer = 3.0 min
    2. Disable supply-aware adjustment
    3. Set risk_lambda = 2.0 (conservative)
    4. Alert on-call engineer
    5. Log all decisions for post-mortem
```

---

## Running the System

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install & Run

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Start server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Run full end-to-end simulation |
| `/kitchen/state` | POST | Estimate kitchen congestion |
| `/kitchen/kpt` | POST | Get KPT distribution |
| `/reliability/mri` | POST | Compute MRI |
| `/reliability/kvi` | POST | Compute KVI |
| `/dispatch/decide` | POST | Make dispatch decision |
| `/dispatch/batch` | POST | Batch dispatch decisions |
| `/dispatch/optimize-lambda` | POST | Monte Carlo λ optimization |
| `/monitoring/metrics` | GET | System health metrics |
| `/monitoring/alerts` | GET | Active guardrail alerts |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

### Example: Dispatch Decision

```bash
curl -X POST http://localhost:8000/dispatch/decide \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD-12345",
    "merchant_id": "merchant_0001",
    "expected_travel_time_minutes": 5.0,
    "nearby_rider_count": 3
  }'
```

Response includes: assignment delay, safety buffer breakdown, risk level, reason codes, and expected cost.
