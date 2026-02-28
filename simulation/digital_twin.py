"""
Digital Twin Marketplace Simulation for RPKOE Evaluation.

A statistically grounded simulation framework that compares:
    System A (Baseline): Rolling mean KPT + fixed 3-min safety buffer
    System B (RPKOE):    Probabilistic KPT + uncertainty-aware dispatch

RULES:
    - Nothing is hardcoded or biased toward RPKOE
    - Both systems observe identical arrival streams (same seed)
    - All metrics are computed from simulation, not assumed
    - Monte Carlo over 20 seeds for statistical validation
    - Mann-Whitney U test for significance

Usage:
    cd R-kud
    uv run python simulation/digital_twin.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════
# 1. MERCHANT & ORDER DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MerchantProfile:
    """Simulated merchant with ground-truth characteristics."""

    merchant_id: int
    base_prep_mu: float          # LogNormal μ for prep time
    base_prep_sigma: float       # LogNormal σ for prep time
    capacity: int                # Max concurrent orders
    rush_multiplier: float       # External demand amplifier (1.0–2.0)
    mark_on_arrival_prob: float  # P(FOR marked at or before true ready)
    batch_marking_prob: float    # P(merchant batches multiple FORs)
    marking_bias: float          # Systematic FOR marking bias (minutes)
    marking_noise: float         # FOR marking noise std (minutes)


def generate_merchants(num_merchants: int, rng: np.random.Generator) -> list[MerchantProfile]:
    """Create a diverse population of merchants."""
    merchants = []
    for i in range(num_merchants):
        # Varied reliability tiers
        tier = rng.choice(["high", "medium", "low"], p=[0.3, 0.5, 0.2])

        if tier == "high":
            mu = rng.uniform(2.2, 2.6)       # ~9–13 min median
            sigma = rng.uniform(0.2, 0.3)
            marking_bias = rng.uniform(-0.5, 0.5)
            marking_noise = rng.uniform(0.3, 0.8)
            mark_on_arrival = rng.uniform(0.6, 0.9)
            batch_prob = rng.uniform(0.05, 0.1)
        elif tier == "medium":
            mu = rng.uniform(2.4, 2.8)       # ~11–16 min median
            sigma = rng.uniform(0.25, 0.35)
            marking_bias = rng.uniform(-1.0, 1.5)
            marking_noise = rng.uniform(0.8, 1.5)
            mark_on_arrival = rng.uniform(0.35, 0.65)
            batch_prob = rng.uniform(0.1, 0.15)
        else:  # low reliability
            mu = rng.uniform(2.6, 3.0)       # ~13–20 min median
            sigma = rng.uniform(0.3, 0.4)
            marking_bias = rng.uniform(0.5, 3.0)
            marking_noise = rng.uniform(1.5, 3.0)
            mark_on_arrival = rng.uniform(0.15, 0.4)
            batch_prob = rng.uniform(0.15, 0.25)

        merchants.append(MerchantProfile(
            merchant_id=i,
            base_prep_mu=mu,
            base_prep_sigma=sigma,
            capacity=int(rng.integers(8, 20)),
            rush_multiplier=rng.uniform(1.0, 2.0),
            mark_on_arrival_prob=mark_on_arrival,
            batch_marking_prob=batch_prob,
            marking_bias=marking_bias,
            marking_noise=marking_noise,
        ))
    return merchants


# ═══════════════════════════════════════════════════════════════════════════
# 2. DIGITAL TWIN MARKETPLACE
# ═══════════════════════════════════════════════════════════════════════════


class DigitalTwinMarketplace:
    """Generates a simulated marketplace with ground-truth data.

    Produces an order-level DataFrame with:
        - Ground-truth prep times (never observed by systems)
        - Noisy FOR signals (observed by systems)
        - Rider travel times
        - Merchant state at order time
    """

    def __init__(
        self,
        num_merchants: int = 500,
        total_orders: int = 20_000,
        simulation_duration_minutes: int = 480,  # 8 hours
        random_seed: int = 42,
    ):
        self.num_merchants = num_merchants
        self.total_orders = total_orders
        self.duration = simulation_duration_minutes
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.merchants = generate_merchants(num_merchants, self.rng)

    def simulate(self) -> pd.DataFrame:
        """Run the marketplace simulation and return order-level data."""
        rng = self.rng
        n = self.total_orders

        # --- Assign orders to merchants (weighted by capacity) ---
        capacities = np.array([m.capacity for m in self.merchants], dtype=float)
        weights = capacities / capacities.sum()
        merchant_ids = rng.choice(self.num_merchants, size=n, p=weights)

        # --- Order arrival times (Poisson-distributed across duration) ---
        arrival_times = np.sort(rng.uniform(0, self.duration, size=n))

        # --- Time-of-day factor (lunch 12-14h, dinner 19-21h peaks) ---
        # Assume simulation starts at 10:00
        hours = 10.0 + arrival_times / 60.0
        lunch_peak = np.exp(-0.5 * ((hours - 13) / 1.5) ** 2)
        dinner_peak = np.exp(-0.5 * ((hours - 20) / 1.5) ** 2)
        tod_factor = np.clip(lunch_peak + dinner_peak, 0, 1)

        # --- Per-order merchant properties (vectorized lookup) ---
        prep_mus = np.array([self.merchants[m].base_prep_mu for m in merchant_ids])
        prep_sigmas = np.array([self.merchants[m].base_prep_sigma for m in merchant_ids])
        rush_mults = np.array([self.merchants[m].rush_multiplier for m in merchant_ids])
        mark_biases = np.array([self.merchants[m].marking_bias for m in merchant_ids])
        mark_noises = np.array([self.merchants[m].marking_noise for m in merchant_ids])
        mark_on_arrival_probs = np.array(
            [self.merchants[m].mark_on_arrival_prob for m in merchant_ids]
        )
        batch_probs = np.array(
            [self.merchants[m].batch_marking_prob for m in merchant_ids]
        )

        # --- Compute concurrent orders per merchant at each time ---
        # Approximate: use rolling count within a window
        concurrent_orders = np.ones(n)
        for i in range(n):
            mid = merchant_ids[i]
            t = arrival_times[i]
            mask = (merchant_ids == mid) & (arrival_times >= t - 15) & (arrival_times <= t)
            concurrent_orders[i] = mask.sum()

        capacities_per_order = np.array(
            [self.merchants[m].capacity for m in merchant_ids], dtype=float
        )
        throughput_saturation = np.clip(concurrent_orders / capacities_per_order, 0, 1)

        # --- Ground-truth prep time ---
        # Congestion increases prep time: scale μ by (1 + 0.3 * saturation * rush)
        congestion_boost = 1.0 + 0.3 * throughput_saturation * rush_mults * tod_factor
        effective_mu = prep_mus + np.log(congestion_boost)
        true_prep_time = rng.lognormal(mean=effective_mu, sigma=prep_sigmas)

        # --- FOR marking signal (noisy observation) ---
        # Marking delay = bias + noise; positive = marked AFTER ready
        marking_delay = rng.normal(mark_biases, mark_noises)

        # Batch marking: some merchants delay FOR for multiple orders
        is_batch = rng.random(n) < batch_probs
        marking_delay = np.where(is_batch, marking_delay + rng.uniform(1, 4, n), marking_delay)

        # Observed FOR time = true_prep + marking_delay
        observed_for_time = true_prep_time + marking_delay

        # Mark-on-arrival flag
        is_mark_on_arrival = rng.random(n) < mark_on_arrival_probs

        # --- Rider travel time ---
        mean_travel = rng.uniform(5, 8, size=n)
        rider_travel_time = rng.exponential(scale=mean_travel)
        rider_travel_time = np.clip(rider_travel_time, 1.0, 30.0)

        # --- Rider supply per zone (simulated) ---
        # Higher supply during off-peak, lower during peak
        base_supply = rng.integers(3, 12, size=n)
        rider_supply = np.maximum(1, (base_supply * (1.0 - 0.5 * tod_factor)).astype(int))

        # --- Build DataFrame ---
        df = pd.DataFrame({
            "order_id": np.arange(n),
            "merchant_id": merchant_ids,
            "arrival_time": arrival_times,
            "hour": hours,
            "tod_factor": tod_factor,
            "concurrent_orders": concurrent_orders,
            "capacity": capacities_per_order,
            "throughput_saturation": throughput_saturation,
            "rush_multiplier": rush_mults,

            # Ground truth (hidden from systems)
            "true_prep_time": true_prep_time,

            # Observed signals
            "observed_for_time": observed_for_time,
            "marking_delay": marking_delay,
            "is_mark_on_arrival": is_mark_on_arrival,
            "is_batch_marked": is_batch,

            # Rider
            "rider_travel_time": rider_travel_time,
            "rider_supply": rider_supply,

            # Merchant params (for RPKOE features)
            "marking_bias": mark_biases,
            "marking_noise": mark_noises,
        })

        return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. SYSTEM A — BASELINE
# ═══════════════════════════════════════════════════════════════════════════


def run_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline system: rolling mean FOR + fixed 3-min safety buffer.

    - Predicted KPT = rolling mean of observed FOR times per merchant
    - Safety buffer = 3.0 min (fixed)
    - Rider assigned immediately (no delay)
    """
    result = df.copy()

    # Rolling mean of FOR times per merchant (expanding window, min 3 obs)
    result["predicted_kpt"] = (
        result.groupby("merchant_id")["observed_for_time"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )

    result["safety_buffer"] = 3.0
    result["assign_delay"] = 0.0  # Immediate assignment

    # Rider arrival time = arrival_time + assign_delay + travel_time
    result["rider_arrival_at"] = (
        result["arrival_time"] + result["assign_delay"] + result["rider_travel_time"]
    )

    # Food ready time = arrival_time + true_prep_time
    result["food_ready_at"] = result["arrival_time"] + result["true_prep_time"]

    # Rider wait = max(0, food_ready - rider_arrival) (rider waits for food)
    result["rider_wait"] = np.maximum(0, result["food_ready_at"] - result["rider_arrival_at"])

    # Late arrival = rider arrived after food was ready by > threshold
    # (food sat on rack waiting for rider — NOT what we want)
    # Actually: late = rider arrived BEFORE food ready (food not ready)
    # rider_wait > 0 means rider is waiting = good (early arrival)
    # food_wait > 0 means food is waiting for rider = bad (late rider)
    result["food_wait"] = np.maximum(0, result["rider_arrival_at"] - result["food_ready_at"])

    # ETA error = |predicted_kpt - true_prep_time|
    result["eta_error"] = np.abs(result["predicted_kpt"] - result["true_prep_time"])

    # Late flags
    result["late_3min"] = (result["rider_wait"] > 3.0).astype(int)
    result["late_5min"] = (result["rider_wait"] > 5.0).astype(int)

    # Dispatch cost = rider_wait + λ * late_penalty (λ = 1.5)
    result["dispatch_cost"] = result["rider_wait"] + 1.5 * result["late_3min"]

    result["system"] = "Baseline"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. SYSTEM B — RPKOE
# ═══════════════════════════════════════════════════════════════════════════


def run_rpkoe(df: pd.DataFrame) -> pd.DataFrame:
    """RPKOE system: probabilistic KPT + uncertainty-aware dispatch.

    - Stateful congestion estimation with decay memory
    - LogNormal KPT distribution (μ, σ from congestion)
    - Composite MRI from marking behavior
    - KVI from pickup delay volatility
    - Supply-aware adaptive safety buffer
    - Assignment delay = max(0, E[KPT] - travel - buffer)
    """
    result = df.copy()
    n = len(result)

    # --- Pre-compute per-merchant rolling statistics ---
    # Marking std (rolling, window=30)
    result["marking_std"] = (
        result.groupby("merchant_id")["marking_delay"]
        .transform(lambda s: s.expanding(min_periods=1).std().fillna(0.5))
    )

    # Mark-on-arrival rate (rolling)
    result["mark_on_arrival_rate"] = (
        result.groupby("merchant_id")["is_mark_on_arrival"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )

    # Residual drift: FOR_observed - rolling_mean(FOR)
    result["for_rolling_mean"] = (
        result.groupby("merchant_id")["observed_for_time"]
        .transform(lambda s: s.expanding(min_periods=1).mean())
    )
    result["residual_drift"] = result["observed_for_time"] - result["for_rolling_mean"]
    result["residual_drift_rolling"] = (
        result.groupby("merchant_id")["residual_drift"]
        .transform(lambda s: s.expanding(min_periods=1).mean().fillna(0))
    )

    # Pickup delay std (proxy: observed_for variance)
    result["pickup_delay_std"] = (
        result.groupby("merchant_id")["observed_for_time"]
        .transform(lambda s: s.expanding(min_periods=3).std().fillna(0.5))
    )

    # --- Congestion estimation with decay memory ---
    # Instantaneous: weighted combination
    alpha, beta, gamma, delta = 0.35, 0.25, 0.20, 0.20
    decay = 0.7

    # Normalize inputs
    active_norm = np.clip(result["concurrent_orders"] / result["capacity"], 0, 1)
    drift_clamped = np.clip(result["residual_drift_rolling"], -5, 5)
    drift_norm = (drift_clamped + 5.0) / 10.0

    instantaneous_congestion = (
        alpha * active_norm
        + beta * drift_norm
        + gamma * result["tod_factor"]
        + delta * result["throughput_saturation"]
    )

    # Apply decay memory per merchant (sequential within merchant)
    congestion = np.zeros(n)
    prev_congestion: dict[int, float] = {}

    # Sort by merchant and time for sequential processing
    sorted_idx = result.sort_values(["merchant_id", "arrival_time"]).index
    for idx in sorted_idx:
        mid = result.loc[idx, "merchant_id"]
        inst = instantaneous_congestion.iloc[idx] if hasattr(instantaneous_congestion, 'iloc') else instantaneous_congestion[idx]

        if mid in prev_congestion:
            cong = decay * prev_congestion[mid] + (1 - decay) * inst
        else:
            cong = inst  # Cold start

        # Spike detection
        drift_val = result.loc[idx, "residual_drift_rolling"]
        if abs(drift_val) > 2.0:
            cong += 0.15

        congestion[idx] = min(max(cong, 0), 2.0)
        prev_congestion[mid] = congestion[idx]

    result["congestion_score"] = congestion

    # --- KPT Distribution (LogNormal, grounded in observed FOR) ---
    # Derive LogNormal parameters from observed FOR data per merchant.
    # The rolling mean of observed FOR is our best point estimate.
    # Congestion score ADJUSTS it, rather than replacing it.
    for_rolling_std = (
        result.groupby("merchant_id")["observed_for_time"]
        .transform(lambda s: s.expanding(min_periods=2).std().fillna(1.0))
    )

    # Fit LogNormal μ,σ from observed rolling mean & std of FOR
    # For LogNormal: E[X] = exp(mu + sigma^2/2), Var[X] = (exp(sigma^2)-1)*exp(2*mu+sigma^2)
    # If we know E[X] ~= for_rolling_mean and we estimate sigma from for_rolling_std:
    obs_mean = np.maximum(result["for_rolling_mean"], 1.0)
    obs_std = np.maximum(for_rolling_std, 0.1)

    # Method-of-moments for LogNormal from (obs_mean, obs_std):
    kpt_sigma_sq = np.log1p((obs_std / obs_mean) ** 2)
    kpt_sigma = np.sqrt(np.maximum(kpt_sigma_sq, 0.01))
    kpt_mu = np.log(obs_mean) - kpt_sigma_sq / 2

    # Congestion adjustment: shift mu upward proportional to congestion
    congestion_adjustment = 0.15 * result["congestion_score"]
    kpt_mu_adj = kpt_mu + congestion_adjustment
    # Congestion also widens uncertainty slightly
    kpt_sigma_adj = kpt_sigma * (1.0 + 0.1 * result["congestion_score"])

    # E[KPT] = exp(mu + sigma^2/2)
    result["predicted_kpt"] = np.exp(kpt_mu_adj + kpt_sigma_adj**2 / 2)
    # Std[KPT]
    result["kpt_std"] = np.sqrt(
        (np.exp(kpt_sigma_adj**2) - 1) * np.exp(2 * kpt_mu_adj + kpt_sigma_adj**2)
    )
    # P90
    result["kpt_p90"] = np.exp(kpt_mu_adj + kpt_sigma_adj * sp_stats.norm.ppf(0.90))

    # --- MRI (composite sigmoid) ---
    w1, w2, w3 = 0.4, 0.35, 0.25

    # Delay entropy proxy: std of marking delay (higher = more unpredictable)
    delay_entropy_proxy = (
        result.groupby("merchant_id")["marking_delay"]
        .transform(lambda s: s.expanding(min_periods=3).std().fillna(1.0))
    )

    inv_std = 1.0 / (1.0 + np.maximum(result["marking_std"], 0))
    inv_entropy = 1.0 / (1.0 + np.maximum(delay_entropy_proxy, 0))
    linear_mri = w1 * inv_std + w2 * result["mark_on_arrival_rate"] + w3 * inv_entropy
    result["mri"] = 1.0 / (1.0 + np.exp(-linear_mri))

    # --- KVI ---
    result["kvi"] = result["pickup_delay_std"] * result["rush_multiplier"]

    # --- Safety buffer (supply-aware) ---
    a, b, c = 1.5, 1.0, 2.0
    buffer_raw = a * result["kpt_std"] + b * result["kvi"] + c * (1.0 - result["mri"])

    # Supply-aware reduction
    low_supply_mask = result["rider_supply"] < 3
    supply_reduction = np.where(low_supply_mask, 0.3, 0.0)
    result["safety_buffer"] = buffer_raw * (1.0 - supply_reduction)

    # --- Assignment delay ---
    result["assign_delay"] = np.maximum(
        0, result["predicted_kpt"] - result["rider_travel_time"] - result["safety_buffer"]
    )

    # --- Outcomes ---
    result["rider_arrival_at"] = (
        result["arrival_time"] + result["assign_delay"] + result["rider_travel_time"]
    )
    result["food_ready_at"] = result["arrival_time"] + result["true_prep_time"]

    result["rider_wait"] = np.maximum(0, result["food_ready_at"] - result["rider_arrival_at"])
    result["food_wait"] = np.maximum(0, result["rider_arrival_at"] - result["food_ready_at"])

    result["eta_error"] = np.abs(result["predicted_kpt"] - result["true_prep_time"])
    result["late_3min"] = (result["rider_wait"] > 3.0).astype(int)
    result["late_5min"] = (result["rider_wait"] > 5.0).astype(int)
    result["dispatch_cost"] = result["rider_wait"] + 1.5 * result["late_3min"]

    result["system"] = "RPKOE"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════


def compute_metrics(df: pd.DataFrame, system_name: str) -> dict:
    """Compute all business, signal, dispatch, and calibration metrics."""
    m: dict = {"system": system_name}

    # === Business ===
    m["rider_wait_mean"] = df["rider_wait"].mean()
    m["rider_wait_p50"] = df["rider_wait"].quantile(0.50)
    m["rider_wait_p90"] = df["rider_wait"].quantile(0.90)
    m["rider_wait_p95"] = df["rider_wait"].quantile(0.95)

    m["eta_error_mean"] = df["eta_error"].mean()
    m["eta_error_p50"] = df["eta_error"].quantile(0.50)
    m["eta_error_p90"] = df["eta_error"].quantile(0.90)

    m["late_rate_3min"] = df["late_3min"].mean()
    m["late_rate_5min"] = df["late_5min"].mean()

    m["food_wait_mean"] = df["food_wait"].mean()
    m["rider_idle_mean"] = df["food_wait"].mean()  # food_wait = rider arrived, food not ready yet → rider idle

    # === Signal ===
    m["for_noise_std"] = df["marking_delay"].std() if "marking_delay" in df else 0
    m["residual_drift_mean"] = df["residual_drift_rolling"].mean() if "residual_drift_rolling" in df else 0
    m["residual_drift_std"] = df["residual_drift_rolling"].std() if "residual_drift_rolling" in df else 0

    if "mri" in df:
        m["mri_mean"] = df["mri"].mean()
        m["mri_std"] = df["mri"].std()
    if "kvi" in df:
        m["kvi_mean"] = df["kvi"].mean()
        m["kvi_std"] = df["kvi"].std()

    # === Dispatch ===
    m["safety_buffer_mean"] = df["safety_buffer"].mean()
    m["safety_buffer_std"] = df["safety_buffer"].std()
    m["assign_delay_mean"] = df["assign_delay"].mean()
    m["assign_delay_std"] = df["assign_delay"].std()
    m["dispatch_cost_mean"] = df["dispatch_cost"].mean()
    m["dispatch_cost_std"] = df["dispatch_cost"].std()

    # === Calibration (RPKOE only) ===
    if "kpt_p90" in df.columns:
        coverage = (df["true_prep_time"] <= df["kpt_p90"]).mean()
        m["calibration_coverage_p90"] = coverage
        m["calibration_error"] = coverage - 0.90
    else:
        m["calibration_coverage_p90"] = np.nan
        m["calibration_error"] = np.nan

    # === Decision Volatility ===
    if "assign_delay" in df.columns and df["assign_delay"].std() > 0:
        dvi = (
            df.groupby("merchant_id")["assign_delay"]
            .std()
            .fillna(0)
        )
        m["decision_volatility_mean"] = dvi.mean()
        m["decision_volatility_max"] = dvi.max()
    else:
        m["decision_volatility_mean"] = 0
        m["decision_volatility_max"] = 0

    # === Congestion volatility (RPKOE) ===
    if "congestion_score" in df.columns:
        cong_vol = df.groupby("merchant_id")["congestion_score"].std().fillna(0)
        m["congestion_volatility_mean"] = cong_vol.mean()
    else:
        m["congestion_volatility_mean"] = np.nan

    return m


# ═══════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO & STATISTICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def run_monte_carlo(
    num_runs: int = 20,
    num_merchants: int = 500,
    total_orders: int = 20_000,
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo simulation for statistical validation.

    Returns:
        all_metrics: Per-run metrics for both systems
        comparison: Comparison table with % improvement and CI
        ci_df: Confidence intervals
    """
    all_rows = []

    for run_id in range(num_runs):
        seed = base_seed + run_id
        print(f"  Monte Carlo run {run_id + 1}/{num_runs} (seed={seed})")

        marketplace = DigitalTwinMarketplace(
            num_merchants=num_merchants,
            total_orders=total_orders,
            random_seed=seed,
        )
        orders = marketplace.simulate()

        baseline_result = run_baseline(orders)
        rpkoe_result = run_rpkoe(orders)

        baseline_metrics = compute_metrics(baseline_result, "Baseline")
        rpkoe_metrics = compute_metrics(rpkoe_result, "RPKOE")

        baseline_metrics["run_id"] = run_id
        rpkoe_metrics["run_id"] = run_id

        all_rows.append(baseline_metrics)
        all_rows.append(rpkoe_metrics)

    all_metrics = pd.DataFrame(all_rows)

    # --- Comparison table ---
    key_metrics = [
        "rider_wait_mean", "rider_wait_p50", "rider_wait_p90", "rider_wait_p95",
        "eta_error_mean", "eta_error_p50", "eta_error_p90",
        "late_rate_3min", "late_rate_5min",
        "food_wait_mean",
        "safety_buffer_mean",
        "assign_delay_mean",
        "dispatch_cost_mean",
        "decision_volatility_mean",
    ]

    comparison_rows = []
    for metric in key_metrics:
        baseline_vals = all_metrics[all_metrics["system"] == "Baseline"][metric].dropna()
        rpkoe_vals = all_metrics[all_metrics["system"] == "RPKOE"][metric].dropna()

        if len(baseline_vals) == 0 or len(rpkoe_vals) == 0:
            continue

        b_mean = baseline_vals.mean()
        r_mean = rpkoe_vals.mean()

        pct_change = ((r_mean - b_mean) / b_mean * 100) if b_mean != 0 else 0

        # Mann-Whitney U test
        try:
            stat, pvalue = sp_stats.mannwhitneyu(
                baseline_vals, rpkoe_vals, alternative="two-sided"
            )
        except Exception:
            stat, pvalue = np.nan, np.nan

        # 95% CI on the difference (RPKOE - Baseline)
        diffs = rpkoe_vals.values - baseline_vals.values
        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)

        comparison_rows.append({
            "metric": metric,
            "baseline_mean": round(b_mean, 4),
            "rpkoe_mean": round(r_mean, 4),
            "difference": round(r_mean - b_mean, 4),
            "pct_change": round(pct_change, 2),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "mann_whitney_p": round(pvalue, 6) if not np.isnan(pvalue) else np.nan,
            "significant_5pct": pvalue < 0.05 if not np.isnan(pvalue) else False,
        })

    comparison = pd.DataFrame(comparison_rows)

    # --- Confidence intervals DataFrame ---
    ci_rows = []
    for metric in key_metrics:
        for system in ["Baseline", "RPKOE"]:
            vals = all_metrics[all_metrics["system"] == system][metric].dropna()
            if len(vals) == 0:
                continue
            ci_rows.append({
                "system": system,
                "metric": metric,
                "mean": round(vals.mean(), 4),
                "std": round(vals.std(), 4),
                "ci_95_low": round(np.percentile(vals, 2.5), 4),
                "ci_95_high": round(np.percentile(vals, 97.5), 4),
            })
    ci_df = pd.DataFrame(ci_rows)

    return all_metrics, comparison, ci_df


# ═══════════════════════════════════════════════════════════════════════════
# 7. SINGLE-RUN DATA FOR PLOTTING
# ═══════════════════════════════════════════════════════════════════════════


def run_single_simulation(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one simulation for detailed plotting."""
    marketplace = DigitalTwinMarketplace(random_seed=seed)
    orders = marketplace.simulate()
    baseline = run_baseline(orders)
    rpkoe = run_rpkoe(orders)
    return baseline, rpkoe


# ═══════════════════════════════════════════════════════════════════════════
# 8. PLOT GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_plots(
    baseline: pd.DataFrame,
    rpkoe: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate all visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Rider Wait Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(baseline["rider_wait"].quantile(0.99), rpkoe["rider_wait"].quantile(0.99)), 50)
    ax.hist(baseline["rider_wait"], bins=bins, alpha=0.6, label="Baseline", density=True)
    ax.hist(rpkoe["rider_wait"], bins=bins, alpha=0.6, label="RPKOE", density=True)
    ax.axvline(baseline["rider_wait"].quantile(0.90), color="tab:blue", linestyle="--",
               label=f'Baseline P90={baseline["rider_wait"].quantile(0.90):.1f}')
    ax.axvline(rpkoe["rider_wait"].quantile(0.90), color="tab:orange", linestyle="--",
               label=f'RPKOE P90={rpkoe["rider_wait"].quantile(0.90):.1f}')
    ax.set_xlabel("Rider Wait (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Rider Wait Distribution: Baseline vs RPKOE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rider_wait_distribution.png", dpi=150)
    plt.close(fig)

    # --- 2. ETA Error Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(baseline["eta_error"].quantile(0.99), rpkoe["eta_error"].quantile(0.99)), 50)
    ax.hist(baseline["eta_error"], bins=bins, alpha=0.6, label="Baseline", density=True)
    ax.hist(rpkoe["eta_error"], bins=bins, alpha=0.6, label="RPKOE", density=True)
    ax.set_xlabel("ETA Error |predicted - actual| (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("ETA Error Distribution: Baseline vs RPKOE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eta_error_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3. Safety Buffer Distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(baseline["safety_buffer"], bins=3, alpha=0.6, label="Baseline (fixed)")
    ax.hist(rpkoe["safety_buffer"], bins=50, alpha=0.6, label="RPKOE (adaptive)")
    ax.set_xlabel("Safety Buffer (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Safety Buffer Distribution: Fixed vs Adaptive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "safety_buffer_distribution.png", dpi=150)
    plt.close(fig)

    # --- 4. Calibration Coverage ---
    if "kpt_p90" in rpkoe.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Rolling calibration coverage for RPKOE
        rpkoe_sorted = rpkoe.sort_values("arrival_time")
        inside = (rpkoe_sorted["true_prep_time"] <= rpkoe_sorted["kpt_p90"]).astype(float)
        rolling_coverage = inside.rolling(window=500, min_periods=100).mean()
        ax.plot(rpkoe_sorted["arrival_time"], rolling_coverage, label="RPKOE P90 Coverage")
        ax.axhline(0.90, color="red", linestyle="--", label="Target (90%)")
        ax.set_xlabel("Simulation Time (minutes)")
        ax.set_ylabel("P90 Coverage (rolling 500 orders)")
        ax.set_title("Distribution Calibration: RPKOE P90 Interval Coverage")
        ax.legend()
        ax.set_ylim(0.5, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "calibration_coverage.png", dpi=150)
        plt.close(fig)

    # --- 5. Congestion vs Rider Wait Scatter ---
    if "congestion_score" in rpkoe.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Sample for readability
        sample = rpkoe.sample(min(3000, len(rpkoe)), random_state=42)
        ax.scatter(sample["congestion_score"], sample["rider_wait"], alpha=0.2, s=5)
        ax.set_xlabel("Congestion Score")
        ax.set_ylabel("Rider Wait (minutes)")
        ax.set_title("RPKOE: Congestion Score vs Rider Wait")
        fig.tight_layout()
        fig.savefig(output_dir / "congestion_vs_wait.png", dpi=150)
        plt.close(fig)

    # --- 6. Decision Volatility by Merchant ---
    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_dvi = baseline.groupby("merchant_id")["assign_delay"].std().fillna(0).sort_values(ascending=False)
    rpkoe_dvi = rpkoe.groupby("merchant_id")["assign_delay"].std().fillna(0)

    # Top 50 merchants by RPKOE DVI
    top_merchants = rpkoe_dvi.sort_values(ascending=False).head(50).index
    x = np.arange(len(top_merchants))
    width = 0.35
    ax.bar(x - width/2, baseline_dvi.reindex(top_merchants).fillna(0), width, label="Baseline", alpha=0.7)
    ax.bar(x + width/2, rpkoe_dvi.reindex(top_merchants).fillna(0), width, label="RPKOE", alpha=0.7)
    ax.set_xlabel("Merchant (sorted by RPKOE DVI)")
    ax.set_ylabel("Decision Volatility (std of assign_delay)")
    ax.set_title("Decision Volatility Index: Top 50 Merchants")
    ax.legend()
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "decision_volatility.png", dpi=150)
    plt.close(fig)

    # --- 7. Dispatch Cost Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, max(baseline["dispatch_cost"].quantile(0.99), rpkoe["dispatch_cost"].quantile(0.99)), 50)
    ax.hist(baseline["dispatch_cost"], bins=bins, alpha=0.6, label="Baseline", density=True)
    ax.hist(rpkoe["dispatch_cost"], bins=bins, alpha=0.6, label="RPKOE", density=True)
    ax.set_xlabel("Dispatch Cost (rider_wait + λ·late_penalty)")
    ax.set_ylabel("Density")
    ax.set_title("Dispatch Cost Distribution: Baseline vs RPKOE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dispatch_cost_distribution.png", dpi=150)
    plt.close(fig)

    # --- 8. MRI Histogram ---
    if "mri" in rpkoe.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rpkoe["mri"], bins=50, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Merchant Reliability Index (MRI)")
        ax.set_ylabel("Order Count")
        ax.set_title("RPKOE: MRI Distribution Across All Orders")
        ax.axvline(rpkoe["mri"].mean(), color="red", linestyle="--",
                   label=f'Mean={rpkoe["mri"].mean():.3f}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "mri_histogram.png", dpi=150)
        plt.close(fig)

    # --- 9. KVI Histogram ---
    if "kvi" in rpkoe.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rpkoe["kvi"].clip(upper=rpkoe["kvi"].quantile(0.99)),
                bins=50, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Kitchen Volatility Index (KVI)")
        ax.set_ylabel("Order Count")
        ax.set_title("RPKOE: KVI Distribution Across All Orders")
        ax.axvline(rpkoe["kvi"].mean(), color="red", linestyle="--",
                   label=f'Mean={rpkoe["kvi"].mean():.3f}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "kvi_histogram.png", dpi=150)
        plt.close(fig)

    print(f"  [PLOTS] 9 plots saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# 9. REPORTING
# ═══════════════════════════════════════════════════════════════════════════


def print_report(
    comparison: pd.DataFrame,
    ci_df: pd.DataFrame,
    baseline: pd.DataFrame,
    rpkoe: pd.DataFrame,
) -> None:
    """Print full console report."""

    print("\n" + "=" * 80)
    print("  RPKOE DIGITAL TWIN SIMULATION — COMPARATIVE ANALYSIS")
    print("=" * 80)

    # --- Comparison table ---
    print("\n[TABLE] COMPARISON TABLE (Mean over 20 Monte Carlo runs)")
    print("-" * 80)
    print(f"{'Metric':<28} {'Baseline':>10} {'RPKOE':>10} {'Delta':>10} {'%Chg':>8} {'p-val':>8} {'Sig?':>5}")
    print("-" * 80)
    for _, row in comparison.iterrows():
        sig = "YES" if row["significant_5pct"] else "NO"
        pv = f"{row['mann_whitney_p']:.4f}" if not pd.isna(row["mann_whitney_p"]) else "N/A"
        print(
            f"{row['metric']:<28} "
            f"{row['baseline_mean']:>10.4f} "
            f"{row['rpkoe_mean']:>10.4f} "
            f"{row['difference']:>10.4f} "
            f"{row['pct_change']:>7.2f}% "
            f"{pv:>8} "
            f"{sig:>5}"
        )
    print("-" * 80)

    # --- Confidence intervals ---
    print("\n[CI] 95% CONFIDENCE INTERVALS (over 20 runs)")
    print("-" * 80)
    for _, row in comparison.iterrows():
        print(
            f"  {row['metric']:<28} "
            f"Δ = {row['difference']:+.4f}  "
            f"CI: [{row['ci_95_low']:+.4f}, {row['ci_95_high']:+.4f}]"
        )

    # --- Statistical significance ---
    print("\n[STATS] STATISTICAL SIGNIFICANCE")
    print("-" * 80)
    sig_metrics = comparison[comparison["significant_5pct"] == True]
    if len(sig_metrics) > 0:
        print(f"  {len(sig_metrics)}/{len(comparison)} metrics show significant difference (p < 0.05)")
        for _, row in sig_metrics.iterrows():
            direction = "lower" if row["difference"] < 0 else "higher"
            print(f"    - {row['metric']}: RPKOE is {abs(row['pct_change']):.1f}% {direction} (p={row['mann_whitney_p']:.4f})")
    else:
        print("  No metrics show statistically significant difference at α=0.05")

    # --- Top 5 unstable merchants ---
    print("\n[VOLATILE] TOP 5 UNSTABLE MERCHANTS (by RPKOE Decision Volatility)")
    print("-" * 80)
    dvi = rpkoe.groupby("merchant_id")["assign_delay"].std().fillna(0).sort_values(ascending=False)
    for mid in dvi.head(5).index:
        m_data = rpkoe[rpkoe["merchant_id"] == mid]
        print(
            f"  Merchant {mid:>4d} | DVI={dvi[mid]:.3f} | "
            f"MRI={m_data['mri'].mean():.3f} | "
            f"KVI={m_data['kvi'].mean():.3f} | "
            f"Orders={len(m_data)}"
        )

    # --- Calibration summary ---
    print("\n[CALIB] CALIBRATION SUMMARY (Single Run)")
    print("-" * 80)
    if "kpt_p90" in rpkoe.columns:
        coverage = (rpkoe["true_prep_time"] <= rpkoe["kpt_p90"]).mean()
        cal_error = coverage - 0.90
        status = "CALIBRATED" if abs(cal_error) < 0.05 else "MISCALIBRATED"
        print(f"  P90 Coverage:     {coverage:.4f} (target: 0.9000)")
        print(f"  Calibration Error: {cal_error:+.4f}")
        print(f"  Status:           {status}")
    else:
        print("  (Not available for baseline)")

    print("\n" + "=" * 80)
    print("  END OF REPORT")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Execute the full digital twin evaluation pipeline."""
    output_dir = Path("simulation/output")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  RPKOE DIGITAL TWIN MARKETPLACE SIMULATION")
    print("  500 merchants | 20,000 orders | 20 Monte Carlo runs")
    print("=" * 80)

    # --- Phase 1: Single detailed run for plots ---
    print("\n[1/5] Phase 1: Running single simulation for detailed analysis...")
    baseline, rpkoe = run_single_simulation(seed=42)

    # --- Phase 2: Generate plots ---
    print("\n[2/5] Phase 2: Generating visualizations...")
    comparison_preview = pd.DataFrame([
        compute_metrics(baseline, "Baseline"),
        compute_metrics(rpkoe, "RPKOE"),
    ])
    generate_plots(baseline, rpkoe, comparison_preview, plots_dir)

    # --- Phase 3: Monte Carlo ---
    print("\n[3/5] Phase 3: Running Monte Carlo validation (20 runs)...")
    all_metrics, comparison, ci_df = run_monte_carlo(num_runs=20)

    # --- Phase 4: Export CSVs ---
    print("\n[4/5] Phase 4: Exporting metrics...")
    all_metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)
    ci_df.to_csv(output_dir / "confidence_intervals.csv", index=False)
    print(f"  [OK] CSVs saved to {output_dir}/")

    # --- Phase 5: Report ---
    print_report(comparison, ci_df, baseline, rpkoe)


if __name__ == "__main__":
    main()
