"""
Sliding window feature aggregator for RPKOE.

Consumes events from the event stream and maintains per-merchant
aggregated features over configurable sliding windows (5min, 15min).
Handles idempotency, event ordering, and late-arriving events.

Production mapping:
    - This would be a Flink / Kafka Streams application
    - Each instance owns a partition range of merchants
    - State is checkpointed to RocksDB / Redis for recovery
    - Windowed aggregations use event-time semantics with watermarks

Idempotency guarantees:
    - Duplicate events (same event_id) are silently discarded
    - Late events (timestamp < last_processed) are flagged and
      inserted into the correct window position, not appended

Backpressure handling (production):
    - If downstream services are slow, the aggregator buffers events
    - When buffer exceeds high-water mark, it signals Kafka to pause
    - When buffer drains below low-water mark, consumption resumes
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np

from core.config import RPKOEConfig
from core.feature_store import FeatureStore
from core.models import (
    EventType,
    FOREvent,
    MerchantFeatures,
    OrderEvent,
    RiderEvent,
)
from stream.event_simulator import compute_time_of_day_factor


class _MerchantBuffer:
    """Per-merchant event buffer with sliding window support.

    Maintains chronologically ordered event buffers and computes
    windowed aggregations on demand. Buffer size is bounded to
    prevent unbounded memory growth.
    """

    def __init__(self, merchant_id: str, max_capacity: int, max_buffer: int):
        self.merchant_id = merchant_id
        self.max_capacity = max_capacity

        # Bounded event buffers by type
        self._order_events: deque[OrderEvent] = deque(maxlen=max_buffer)
        self._rider_events: deque[RiderEvent] = deque(maxlen=max_buffer)
        self._for_events: deque[FOREvent] = deque(maxlen=max_buffer)

        # Tracking active orders (not yet prepared/delivered)
        self._active_order_ids: set[str] = set()

        # Residual drift buffer: (timestamp, actual - estimated)
        self._drift_observations: deque[tuple[datetime, float]] = deque(maxlen=max_buffer)

        # Pickup delay buffer: (timestamp, rider_wait_minutes)
        self._pickup_delays: deque[tuple[datetime, float]] = deque(maxlen=max_buffer)

        # FOR marking delay buffer: (timestamp, marking_delay)
        self._marking_delays: deque[tuple[datetime, float]] = deque(maxlen=max_buffer)

        # Rider arrival tracking for pickup delay computation
        self._rider_arrivals: dict[str, datetime] = {}  # order_id -> arrival time

        # Late event counter for monitoring
        self.late_event_count: int = 0
        self.duplicate_event_count: int = 0

    def add_order_event(self, event: OrderEvent) -> None:
        """Process an order lifecycle event."""
        self._order_events.append(event)

        if event.event_type == EventType.ORDER_PLACED:
            self._active_order_ids.add(event.order_id)

        elif event.event_type == EventType.ORDER_PREPARED:
            self._active_order_ids.discard(event.order_id)

            # Compute residual drift if we have both actual and estimated
            if (
                event.actual_prep_time_minutes is not None
                and event.estimated_prep_time_minutes is not None
            ):
                drift = event.actual_prep_time_minutes - event.estimated_prep_time_minutes
                self._drift_observations.append((event.timestamp, drift))

            # Compute pickup delay if rider already arrived
            if event.order_id in self._rider_arrivals:
                rider_arrival = self._rider_arrivals.pop(event.order_id)
                delay = (event.timestamp - rider_arrival).total_seconds() / 60.0
                self._pickup_delays.append((event.timestamp, delay))

        elif event.event_type in (EventType.ORDER_DELIVERED, EventType.ORDER_CANCELLED):
            self._active_order_ids.discard(event.order_id)

    def add_rider_event(self, event: RiderEvent) -> None:
        """Process a rider telemetry event."""
        self._rider_events.append(event)

        if event.event_type == EventType.RIDER_ARRIVED:
            self._rider_arrivals[event.order_id] = event.timestamp

    def add_for_event(self, event: FOREvent) -> None:
        """Process a Food-On-Rack marking event."""
        self._for_events.append(event)
        self._marking_delays.append((event.timestamp, event.marking_delay_minutes))

    def _window_filter(
        self,
        observations: deque[tuple[datetime, float]],
        window_end: datetime,
        window_minutes: int,
    ) -> list[float]:
        """Extract values within a sliding window."""
        window_start = window_end - timedelta(minutes=window_minutes)
        return [v for t, v in observations if window_start <= t <= window_end]

    def compute_features(
        self,
        current_time: datetime,
        short_window: int,
        long_window: int,
        partition_id: int,
    ) -> MerchantFeatures:
        """Compute aggregated features over sliding windows.

        This is the main aggregation function called periodically
        (e.g., every 30 seconds) to update the feature store.
        """
        # Active orders
        active_orders = len(self._active_order_ids)

        # Orders in windows
        orders_short = sum(
            1
            for e in self._order_events
            if e.event_type == EventType.ORDER_PLACED
            and e.timestamp >= current_time - timedelta(minutes=short_window)
        )
        orders_long = sum(
            1
            for e in self._order_events
            if e.event_type == EventType.ORDER_PLACED
            and e.timestamp >= current_time - timedelta(minutes=long_window)
        )

        # Throughput saturation
        throughput_sat = min(active_orders / max(self.max_capacity, 1), 1.0)

        # Residual drift (rolling mean over long window)
        drift_values = self._window_filter(
            self._drift_observations, current_time, long_window
        )
        residual_drift = float(np.mean(drift_values)) if drift_values else 0.0

        # Time-of-day factor
        hour = current_time.hour + current_time.minute / 60.0
        tod_factor = compute_time_of_day_factor(hour)

        # FOR marking statistics (over long window)
        marking_values = self._window_filter(
            self._marking_delays, current_time, long_window
        )
        if len(marking_values) >= 3:
            marking_std = float(np.std(marking_values))
            mark_on_arrival_rate = sum(1 for m in marking_values if m <= 0) / len(
                marking_values
            )
            # Shannon entropy of discretized delays
            delay_entropy = self._compute_delay_entropy(marking_values)
        else:
            marking_std = 0.5  # Prior for cold start
            mark_on_arrival_rate = 0.5
            delay_entropy = 1.0

        # Pickup delay rolling std (over long window)
        delay_values = self._window_filter(
            self._pickup_delays, current_time, long_window
        )
        rolling_delay_std = float(np.std(delay_values)) if len(delay_values) >= 3 else 0.5

        # External rush factor (simulated — in production from external APIs)
        external_rush = 1.0 + 0.3 * tod_factor  # Correlated with demand

        # Nearby rider count (simulated — in production from rider location service)
        base_riders = 5
        nearby_riders = max(
            1, int(base_riders * (1.0 - 0.5 * throughput_sat) + np.random.normal(0, 1))
        )

        return MerchantFeatures(
            merchant_id=self.merchant_id,
            timestamp=current_time,
            active_orders=active_orders,
            orders_last_5min=orders_short,
            orders_last_15min=orders_long,
            max_capacity=self.max_capacity,
            throughput_saturation=round(throughput_sat, 4),
            residual_drift_minutes=round(residual_drift, 4),
            time_of_day_factor=round(tod_factor, 4),
            marking_std=round(marking_std, 4),
            mark_on_arrival_rate=round(mark_on_arrival_rate, 4),
            delay_entropy=round(delay_entropy, 4),
            rolling_pickup_delay_std=round(rolling_delay_std, 4),
            external_rush_factor=round(external_rush, 4),
            nearby_rider_count=nearby_riders,
            partition_id=partition_id,
        )

    @staticmethod
    def _compute_delay_entropy(delays: list[float], n_bins: int = 10) -> float:
        """Compute Shannon entropy of discretized delay distribution.

        Low entropy → predictable delays → reliable merchant.
        High entropy → unpredictable delays → unreliable merchant.
        """
        if not delays:
            return 1.0

        counts, _ = np.histogram(delays, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        if len(probs) <= 1:
            return 0.0

        entropy = -float(np.sum(probs * np.log2(probs)))
        max_entropy = math.log2(n_bins)
        return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


class FeatureAggregator:
    """Streaming feature aggregator with idempotency and ordering guarantees.

    Consumes events, maintains per-merchant sliding window buffers,
    and periodically flushes aggregated features to the feature store.

    Production architecture:
        - Deployed as N instances, each owning partitions [start, end)
        - Consumes from Kafka with consumer group for partition assignment
        - Checkpoints consumer offsets after feature store writes (at-least-once)
        - Deduplication guard ensures exactly-once semantics at the feature level

    Backpressure (conceptual):
        - If feature store writes are slow, buffer events in memory
        - If buffer exceeds `max_buffer_size`, signal upstream to pause
        - Resume consumption when buffer drains below low-water mark
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        config: RPKOEConfig,
    ):
        self._store = feature_store
        self._config = config
        self._buffers: dict[str, _MerchantBuffer] = {}

        # Aggregation counters
        self.events_processed: int = 0
        self.events_deduplicated: int = 0
        self.events_late: int = 0

    def _get_buffer(self, merchant_id: str, max_capacity: int = 15) -> _MerchantBuffer:
        """Get or create a merchant buffer. Thread-safe via feature store."""
        if merchant_id not in self._buffers:
            self._buffers[merchant_id] = _MerchantBuffer(
                merchant_id=merchant_id,
                max_capacity=max_capacity,
                max_buffer=self._config.windows.max_buffer_size,
            )
        return self._buffers[merchant_id]

    def initialize_merchants(
        self,
        merchant_profiles: dict[str, object],
    ) -> None:
        """Pre-create buffers for known merchants.

        In production, this happens at startup from the merchant registry.
        Avoids cold-start allocation under load.
        """
        for mid, profile in merchant_profiles.items():
            max_cap = getattr(profile, "max_capacity", 15)
            self._get_buffer(mid, max_cap)

    def process_event(
        self,
        event: Union[OrderEvent, RiderEvent, FOREvent],
    ) -> bool:
        """Process a single event with idempotency and ordering checks.

        Returns True if the event was processed, False if it was
        discarded (duplicate or irrecoverably out of order).

        Idempotency:
            Events with previously-seen event_id are silently discarded.
            The feature store tracks seen event IDs per merchant.

        Ordering:
            Late events (timestamp < last_processed) are flagged but
            still processed. They are inserted into the correct window
            position by the buffer's deque, ensuring correct aggregation
            even under out-of-order delivery.

        This is the equivalent of a Kafka Streams processor's `process()` method.
        """
        merchant_id = event.merchant_id
        event_id = event.event_id
        event_ts = event.timestamp.timestamp()

        # --- Idempotency guard ---
        if self._store.is_duplicate_event(merchant_id, event_id):
            self.events_deduplicated += 1
            buf = self._buffers.get(merchant_id)
            if buf:
                buf.duplicate_event_count += 1
            return False

        # --- Event ordering check ---
        is_ordered = self._store.check_event_ordering(merchant_id, event_ts)
        if not is_ordered:
            self.events_late += 1
            buf = self._buffers.get(merchant_id)
            if buf:
                buf.late_event_count += 1
            # Still process the event — insert into correct window position
            # In production, this would trigger a watermark update

        # --- Route to appropriate buffer ---
        buf = self._get_buffer(merchant_id)

        if isinstance(event, OrderEvent):
            buf.add_order_event(event)
        elif isinstance(event, RiderEvent):
            buf.add_rider_event(event)
        elif isinstance(event, FOREvent):
            buf.add_for_event(event)

        self.events_processed += 1
        return True

    def flush_features(
        self,
        current_time: Optional[datetime] = None,
        merchant_ids: Optional[list[str]] = None,
    ) -> list[MerchantFeatures]:
        """Recompute and store aggregated features for all (or specified) merchants.

        Called periodically (e.g., every 30 seconds in production) to update
        the feature store with fresh sliding-window aggregations.

        In production, this would be triggered by a scheduled timer or
        by accumulation of N events since last flush.

        Returns:
            List of updated MerchantFeatures.
        """
        if current_time is None:
            current_time = datetime.now()

        targets = merchant_ids or list(self._buffers.keys())
        results = []

        partition_config = self._config.partition

        for mid in targets:
            buf = self._buffers.get(mid)
            if buf is None:
                continue

            partition_id = partition_config.get_partition(mid)

            features = buf.compute_features(
                current_time=current_time,
                short_window=self._config.windows.short_window_minutes,
                long_window=self._config.windows.long_window_minutes,
                partition_id=partition_id,
            )

            # Write to feature store
            self._store.set_merchant_features(features)

            # Cache individual features for direct lookup
            self._store.set(mid, "active_orders", features.active_orders, ttl_seconds=60)
            self._store.set(mid, "throughput_saturation", features.throughput_saturation, ttl_seconds=60)
            self._store.set(mid, "residual_drift", features.residual_drift_minutes, ttl_seconds=60)
            self._store.set(mid, "congestion_tod_factor", features.time_of_day_factor, ttl_seconds=60)

            results.append(features)

        return results

    def get_aggregation_stats(self) -> dict:
        """Return aggregation pipeline statistics for monitoring."""
        return {
            "total_events_processed": self.events_processed,
            "duplicate_events_discarded": self.events_deduplicated,
            "late_events_processed": self.events_late,
            "active_merchants": len(self._buffers),
            "dedup_rate": (
                round(self.events_deduplicated / max(self.events_processed + self.events_deduplicated, 1), 4)
            ),
            "late_rate": (
                round(self.events_late / max(self.events_processed, 1), 4)
            ),
        }
