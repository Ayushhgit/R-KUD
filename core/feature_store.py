"""
In-memory feature store simulating Redis with TTL, partitioning, and thread safety.

Production notes:
    - In production, this would be a Redis Cluster with:
        - Consistent hashing for key distribution across nodes
        - Sorted sets for sliding window data
        - TTL-based automatic expiry
        - Pipeline/batch reads for latency reduction
    - The in-memory implementation mirrors the API surface exactly,
      making migration to Redis a config change, not a code change.

Partitioning strategy:
    - Keys are prefixed with partition_id derived from hash(merchant_id) % N
    - Each partition can be independently scaled in production
    - Feature aggregator workers own non-overlapping partition ranges
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Optional

from core.config import PartitionConfig
from core.models import MerchantFeatures


class FeatureStore:
    """Thread-safe, TTL-aware, partition-aware in-memory feature store.

    Simulates Redis cluster behavior for development and testing.
    All methods use the same key semantics as Redis:
        - Keys: "{partition}:{merchant_id}:{feature_key}"
        - Values: Any serializable Python object
        - TTL: Seconds until automatic expiry

    Thread safety is achieved via a per-partition lock strategy,
    minimizing contention under concurrent access.
    """

    def __init__(self, partition_config: Optional[PartitionConfig] = None):
        self._partition_config = partition_config or PartitionConfig()
        self._num_partitions = self._partition_config.num_partitions

        # Per-partition storage: {partition_id: {key: (value, expiry_timestamp)}}
        self._stores: dict[int, dict[str, tuple[Any, Optional[float]]]] = {
            i: {} for i in range(self._num_partitions)
        }

        # Per-partition locks for fine-grained thread safety
        self._locks: dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(self._num_partitions)
        }

        # Merchant features cache: {merchant_id: MerchantFeatures}
        self._merchant_features: dict[str, MerchantFeatures] = {}
        self._features_lock = threading.Lock()

        # Congestion state memory: {merchant_id: last_congestion_score}
        self._congestion_memory: dict[str, float] = {}
        self._congestion_lock = threading.Lock()

        # Event deduplication: {merchant_id: set(event_id)}
        self._seen_events: dict[str, set[str]] = defaultdict(set)
        self._seen_events_lock = threading.Lock()

        # Event ordering: {merchant_id: last_processed_timestamp}
        self._last_timestamps: dict[str, float] = {}
        self._timestamps_lock = threading.Lock()

    # -------------------------------------------------------------------
    # Core Key-Value Operations
    # -------------------------------------------------------------------

    def _get_partition(self, merchant_id: str) -> int:
        """Deterministic partition assignment."""
        return self._partition_config.get_partition(merchant_id)

    def _make_key(self, merchant_id: str, feature_key: str) -> str:
        """Build a partitioned key."""
        partition = self._get_partition(merchant_id)
        return f"{partition}:{merchant_id}:{feature_key}"

    def set(
        self,
        merchant_id: str,
        feature_key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store a feature value with optional TTL.

        Args:
            merchant_id: Restaurant identifier.
            feature_key: Feature name (e.g., "active_orders", "congestion_score").
            value: Feature value.
            ttl_seconds: Time-to-live in seconds. None = no expiry.
        """
        partition = self._get_partition(merchant_id)
        key = self._make_key(merchant_id, feature_key)
        expiry = time.time() + ttl_seconds if ttl_seconds else None

        with self._locks[partition]:
            self._stores[partition][key] = (value, expiry)

    def get(
        self,
        merchant_id: str,
        feature_key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a feature value, respecting TTL expiry.

        Returns default if key is missing or expired.
        Expired keys are lazily cleaned up on access.
        """
        partition = self._get_partition(merchant_id)
        key = self._make_key(merchant_id, feature_key)

        with self._locks[partition]:
            entry = self._stores[partition].get(key)
            if entry is None:
                return default

            value, expiry = entry
            if expiry is not None and time.time() > expiry:
                # Lazy TTL cleanup
                del self._stores[partition][key]
                return default

            return value

    def delete(self, merchant_id: str, feature_key: str) -> bool:
        """Remove a feature key. Returns True if the key existed."""
        partition = self._get_partition(merchant_id)
        key = self._make_key(merchant_id, feature_key)

        with self._locks[partition]:
            if key in self._stores[partition]:
                del self._stores[partition][key]
                return True
            return False

    # -------------------------------------------------------------------
    # Merchant Features (Aggregated View)
    # -------------------------------------------------------------------

    def set_merchant_features(self, features: MerchantFeatures) -> None:
        """Cache the latest aggregated features for a merchant."""
        with self._features_lock:
            self._merchant_features[features.merchant_id] = features

    def get_merchant_features(self, merchant_id: str) -> Optional[MerchantFeatures]:
        """Retrieve cached merchant features."""
        with self._features_lock:
            return self._merchant_features.get(merchant_id)

    def get_all_merchant_ids(self) -> list[str]:
        """List all merchants with cached features."""
        with self._features_lock:
            return list(self._merchant_features.keys())

    # -------------------------------------------------------------------
    # Congestion State Memory
    # -------------------------------------------------------------------

    def get_previous_congestion(self, merchant_id: str) -> Optional[float]:
        """Retrieve the last congestion score for stateful estimation."""
        with self._congestion_lock:
            return self._congestion_memory.get(merchant_id)

    def set_previous_congestion(self, merchant_id: str, score: float) -> None:
        """Store the latest congestion score for decay-based estimation."""
        with self._congestion_lock:
            self._congestion_memory[merchant_id] = score

    # -------------------------------------------------------------------
    # Idempotency & Event Ordering
    # -------------------------------------------------------------------

    def is_duplicate_event(self, merchant_id: str, event_id: str) -> bool:
        """Check if an event has already been processed (idempotency guard).

        In production, this would use Redis SET with NX flag or a
        Bloom filter for memory-efficient deduplication.
        """
        with self._seen_events_lock:
            if event_id in self._seen_events[merchant_id]:
                return True
            self._seen_events[merchant_id].add(event_id)

            # Prevent unbounded growth: trim to last 10K events
            if len(self._seen_events[merchant_id]) > 10_000:
                # Keep only recent half (FIFO approximation)
                events = list(self._seen_events[merchant_id])
                self._seen_events[merchant_id] = set(events[5_000:])

            return False

    def check_event_ordering(
        self, merchant_id: str, event_timestamp: float
    ) -> bool:
        """Check if an event is in-order.

        Returns True if the event is in-order (should be processed).
        Returns False if the event is late (timestamp < last processed).

        Late events are not discarded — they are flagged for special handling
        by the feature aggregator (e.g., inserted into the correct window
        position rather than appended).
        """
        with self._timestamps_lock:
            last_ts = self._last_timestamps.get(merchant_id, 0.0)
            is_ordered = event_timestamp >= last_ts
            if is_ordered:
                self._last_timestamps[merchant_id] = event_timestamp
            return is_ordered

    # -------------------------------------------------------------------
    # Partition Inspection (Operational)
    # -------------------------------------------------------------------

    def get_partition_stats(self) -> dict[int, int]:
        """Return key count per partition for load monitoring."""
        stats = {}
        for pid in range(self._num_partitions):
            with self._locks[pid]:
                stats[pid] = len(self._stores[pid])
        return stats

    def get_total_keys(self) -> int:
        """Total number of active (non-expired) keys across all partitions."""
        total = 0
        now = time.time()
        for pid in range(self._num_partitions):
            with self._locks[pid]:
                for _key, (_, expiry) in self._stores[pid].items():
                    if expiry is None or now < expiry:
                        total += 1
        return total

    def flush_partition(self, partition_id: int) -> int:
        """Clear all keys in a partition. Returns number of keys removed.

        In production, this would be used during partition rebalancing.
        """
        with self._locks[partition_id]:
            count = len(self._stores[partition_id])
            self._stores[partition_id].clear()
            return count

    def flush_all(self) -> None:
        """Clear all data. Used for testing."""
        for pid in range(self._num_partitions):
            with self._locks[pid]:
                self._stores[pid].clear()
        with self._features_lock:
            self._merchant_features.clear()
        with self._congestion_lock:
            self._congestion_memory.clear()
        with self._seen_events_lock:
            self._seen_events.clear()
        with self._timestamps_lock:
            self._last_timestamps.clear()
