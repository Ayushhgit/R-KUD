# RPKOE — Real-Time Probabilistic Kitchen Orchestration Engine

> Production-grade system prototype for real-time kitchen state estimation and uncertainty-aware rider dispatch optimization in a food delivery marketplace.

RPKOE reframes Kitchen Prep Time prediction from a **static forecasting problem** to a **dynamic state estimation and control problem under partial observability**. The system treats kitchens as stochastic systems with hidden congestion states and optimizes rider dispatch decisions using uncertainty-aware probabilistic modeling.

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Code Structure](#code-structure)
- [Scalability Strategy](#scalability-strategy)
- [Monitoring & Guardrails](#monitoring--guardrails)
- [Computational Complexity](#computational-complexity)
- [Merchant Behavior Feedback Loop](#merchant-behavior-feedback-loop)
- [Pilot & Rollout Plan](#pilot--rollout-plan)
- [Running the System](#running-the-system)
- [Expected Business Impact](#expected-business-impact)

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

> **Conceptual Foundation:** The system performs implicit Bayesian belief updating over the hidden kitchen congestion state using incoming event signals (order load, residual drift, pickup delay). This approximates a state-space model where **P(c_t | observations_≤t)** is continuously updated in a streaming manner. The exponential decay memory and spike detection serve as a computationally efficient proxy for full posterior inference.

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

**Supply-Constrained Dispatch Context:**

Dispatch decisions are made within a constrained supply optimization context. Let S_t be expected rider availability in the next Δ minutes:

```
dispatch_risk ∝ 1 / S_t
```

Low supply increases tolerance for rider wait to avoid unfulfilled orders. The dispatch engine dynamically adjusts aggressiveness based on local rider density.

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

### 7. Risk Calibration & Cost Alignment

The λ parameter in the dispatch objective is **not static**. It is calibrated using:

- **Historical cancellation penalty cost** — Revenue lost per cancelled order
- **Rider idle cost per minute** — Opportunity cost of rider waiting at merchant
- **SLA breach penalty** — Contractual or customer-experience cost of late delivery
- **Customer churn risk modeling** — Long-term LTV impact of repeated poor experiences

λ is optimized per merchant cluster or city zone using offline Monte Carlo simulation over historical order logs (see `/dispatch/optimize-lambda` endpoint). This ties the mathematical abstraction directly to measurable business cost.

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
| **Decision Volatility** | **< 1.5** | **> 2.0** | **Investigate model instability** |

### Decision Volatility Index (DVI)

A novel operational metric that detects model instability:

```
DVI(merchant) = std(assign_delay) over rolling 15-min window
```

If dispatch decisions for the same merchant fluctuate excessively:
- **DVI > 2.0** → Congestion estimation noise or feature store staleness
- **DVI > 3.0** → Possible oscillation in decay memory; trigger coefficient review

This catches failure modes that other metrics miss — a merchant can have normal P90 wait times while the *decision process* is unstable.

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

# TYPE rpkoe_decision_volatility gauge
rpkoe_decision_volatility{merchant="merchant_0042"} 1.23
rpkoe_decision_volatility{merchant="merchant_0087"} 3.41

# TYPE rpkoe_mri_score histogram
rpkoe_mri_score_bucket{le="0.3"} 45
rpkoe_mri_score_bucket{le="0.5"} 123
rpkoe_mri_score_bucket{le="0.7"} 890
rpkoe_mri_score_bucket{le="1.0"} 1200
```

---

## Computational Complexity

| Component | Time Complexity | Notes |
|-----------|----------------|-------|
| Feature update (per event) | **O(1)** | Deque append + counter update |
| Sliding window aggregation | **O(W)** | W = window size (bounded, ≤ 500) |
| Congestion update | **O(1)** | Weighted sum + decay |
| KPT distribution | **O(1)** | Closed-form LogNormal parameterization |
| MRI computation | **O(1)** | Sigmoid over 3 features |
| KVI computation | **O(1)** | Single multiplication |
| Safety buffer | **O(1)** | Weighted sum |
| Dispatch decision | **O(1)** | Composition of above |
| Monte Carlo tuning | **O(K·N)** | K = λ candidates, N = samples (offline) |
| Calibration check | **O(W)** | W = calibration window (bounded) |

**All online decisions are constant-time** and scale linearly with number of merchants via partitioning. The system processes each dispatch decision independently with no cross-merchant dependencies, enabling embarrassingly parallel execution across partition owners.

---

## Merchant Behavior Feedback Loop

The system creates a closed-loop incentive alignment mechanism:

Merchants with persistently **low MRI** (unreliable FOR marking) trigger:

1. **App-level nudges** — Notifications about inconsistent FOR marking accuracy
2. **Auto-adjusted dispatch conservatism** — Higher safety buffers increase rider costs, which can be surfaced to merchants as a "reliability score"
3. **Discovery ranking impact** — Unreliable merchants may receive lower placement in search results, creating economic incentive to improve
4. **Onboarding coaching** — New merchants with degrading MRI during first 30 days receive proactive support

This creates a **virtuous cycle**: better FOR signals → better KPT predictions → lower rider wait → better customer experience → higher order volume for the merchant.

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

---

## Expected Business Impact

Based on simulation dynamics and marketplace operational benchmarks:

| Metric | Projected Improvement | Mechanism |
|--------|----------------------|----------|
| Rider Wait P90 | **8–15% reduction** | Uncertainty-aware dispatch timing |
| ETA Variance | **5–10% reduction** | Probabilistic KPT replaces point estimates |
| Peak-Hour Cancellations | **3–5% reduction** | Adaptive safety buffer during congestion |
| Rider Utilization | **4–8% increase** | Supply-aware buffer reduces unnecessary wait |
| Calibration Stability | **Sustained P90 coverage > 88%** | Continuous drift monitoring + recalibration |
| Merchant FOR Accuracy | **Gradual improvement** | Feedback loop incentivizes consistent marking |

These projections are conservative estimates. The Monte Carlo offline optimizer enables per-cluster parameter tuning that can further improve outcomes in specific city zones and cuisine categories.

---

> *RPKOE is not a forecasting model. It is **marketplace control infrastructure** — a system that senses, estimates, decides, and monitors in a continuous loop under uncertainty.*
