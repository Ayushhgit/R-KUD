"""
Event stream simulator for RPKOE.

Simulates a Kafka-partitioned event stream producing order, rider, and FOR
events with realistic characteristics:
    - Time-of-day demand patterns (lunch 12-14h, dinner 19-21h)
    - Per-merchant capacity and reliability variations
    - Noisy FOR marking delays (some merchants mark early, some late)
    - Rider arrival jitter based on distance and traffic
    - Occasional out-of-order and duplicate events (for idempotency testing)

Production mapping:
    In a real system, this module would be replaced by Kafka consumers
    reading from topics: `orders.events`, `riders.telemetry`, `merchants.for_signals`.
    The event generator interface (async generator) matches the consumption
    pattern of a Kafka consumer group.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator

import numpy as np

from core.config import RPKOEConfig
from core.models import EventType, FOREvent, OrderEvent, RiderEvent


class MerchantProfile:
    """Simulated merchant with persistent characteristics.

    Each merchant has a reliability profile that determines:
    - How accurate their FOR marking is
    - Their kitchen throughput capacity
    - Their typical prep time distribution
    """

    def __init__(
        self,
        merchant_id: str,
        max_capacity: int,
        base_prep_time: float,
        marking_bias: float,
        marking_noise: float,
        rng: np.random.Generator,
    ):
        self.merchant_id = merchant_id
        self.max_capacity = max_capacity
        self.base_prep_time = base_prep_time
        self.marking_bias = marking_bias      # Systematic early/late marking
        self.marking_noise = marking_noise     # Volatility in marking
        self.active_orders: list[dict] = []
        self._rng = rng

    def get_actual_prep_time(self, congestion_factor: float) -> float:
        """Sample actual prep time based on current congestion."""
        base = self.base_prep_time * (1 + 0.5 * congestion_factor)
        noise = self._rng.normal(0, base * 0.2)
        return max(2.0, base + noise)

    def get_marking_delay(self) -> float:
        """Sample FOR marking delay (positive = marked after food ready)."""
        return self._rng.normal(self.marking_bias, self.marking_noise)


def compute_time_of_day_factor(hour: float) -> float:
    """Compute demand multiplier based on time of day.

    Models two peaks:
        - Lunch: peak at 13:00 (hour=13)
        - Dinner: peak at 20:00 (hour=20)

    Uses sum of two Gaussian bumps normalized to [0, 1].
    """
    lunch_peak = np.exp(-0.5 * ((hour - 13) / 1.5) ** 2)
    dinner_peak = np.exp(-0.5 * ((hour - 20) / 1.5) ** 2)
    factor = lunch_peak + dinner_peak
    return float(min(factor, 1.0))


class EventSimulator:
    """Kafka-like event stream generator for the RPKOE prototype.

    Produces a correlated stream of OrderEvent, RiderEvent, and FOREvent
    objects that simulate realistic marketplace dynamics. Events are
    generated in chronological order with configurable noise.

    Usage:
        simulator = EventSimulator(config)
        async for event in simulator.stream():
            process(event)

    Production equivalent:
        kafka_consumer = KafkaConsumer("orders.events", "riders.telemetry")
        for message in kafka_consumer:
            event = deserialize(message)
            process(event)
    """

    def __init__(
        self,
        config: RPKOEConfig,
        rng_seed: int = 42,
        start_hour: float = 11.0,
    ):
        self._config = config
        self._rng = np.random.default_rng(rng_seed)
        self._start_hour = start_hour
        self._merchants: dict[str, MerchantProfile] = {}
        self._initialize_merchants()

    def _initialize_merchants(self) -> None:
        """Create merchant profiles with varied characteristics.

        In production, merchant profiles come from the merchant database.
        Here we simulate a distribution of reliable and unreliable merchants.
        """
        sim = self._config.simulation
        for i in range(sim.num_merchants):
            merchant_id = f"merchant_{i:04d}"

            # Varied reliability: some merchants are accurate, some are noisy
            reliability_class = self._rng.choice(
                ["high", "medium", "low"],
                p=[0.3, 0.5, 0.2],
            )

            if reliability_class == "high":
                marking_bias = self._rng.uniform(-0.5, 0.5)
                marking_noise = self._rng.uniform(0.3, 0.8)
                base_prep = self._rng.uniform(8, 15)
            elif reliability_class == "medium":
                marking_bias = self._rng.uniform(-1.0, 1.5)
                marking_noise = self._rng.uniform(0.8, 1.5)
                base_prep = self._rng.uniform(10, 20)
            else:  # low
                marking_bias = self._rng.uniform(0.5, 3.0)
                marking_noise = self._rng.uniform(1.5, 3.0)
                base_prep = self._rng.uniform(15, 30)

            self._merchants[merchant_id] = MerchantProfile(
                merchant_id=merchant_id,
                max_capacity=self._rng.integers(
                    sim.merchant_max_capacity // 2,
                    sim.merchant_max_capacity + 5,
                ),
                base_prep_time=base_prep,
                marking_bias=marking_bias,
                marking_noise=marking_noise,
                rng=self._rng,
            )

    def get_merchant_profiles(self) -> dict[str, MerchantProfile]:
        """Expose merchant profiles for feature aggregation initialization."""
        return self._merchants

    async def stream(
        self,
        duration_minutes: int | None = None,
        inject_duplicates: bool = True,
        inject_late_events: bool = True,
    ) -> AsyncGenerator[OrderEvent | RiderEvent | FOREvent, None]:
        """Generate a stream of marketplace events.

        Simulates the full lifecycle of orders:
            1. ORDER_PLACED — new order arrives
            2. ORDER_ACCEPTED — merchant accepts
            3. RIDER_ASSIGNED — system assigns a rider
            4. RIDER_ARRIVED — rider reaches the merchant
            5. ORDER_PREPARED — FOR signal (noisy)
            6. RIDER_PICKED_UP — rider picks up food
            7. ORDER_DELIVERED — order complete

        Events are yielded in approximate chronological order, with
        occasional out-of-order events to test ordering logic.

        Args:
            duration_minutes: Simulation length. Defaults to config value.
            inject_duplicates: If True, ~2% of events are duplicated.
            inject_late_events: If True, ~3% of events arrive late.
        """
        sim = self._config.simulation
        duration = duration_minutes or sim.simulation_duration_minutes

        base_time = datetime.now()
        current_minute = 0.0
        pending_events: list[tuple[float, OrderEvent | RiderEvent | FOREvent]] = []

        while current_minute < duration:
            current_hour = self._start_hour + current_minute / 60.0
            tod_factor = compute_time_of_day_factor(current_hour % 24)

            # Determine how many new orders arrive this minute
            for merchant_id, merchant in self._merchants.items():
                # Order rate scales with time-of-day
                order_rate = sim.base_order_rate_per_minute * (0.5 + 1.5 * tod_factor)

                # Generate orders via Poisson process
                n_new_orders = self._rng.poisson(order_rate)

                for _ in range(
                    min(n_new_orders, merchant.max_capacity - len(merchant.active_orders))
                ):
                    order_id = str(uuid.uuid4())[:8]
                    order_time = base_time + timedelta(minutes=current_minute)
                    congestion = len(merchant.active_orders) / max(merchant.max_capacity, 1)

                    # Estimated and actual prep times
                    actual_prep = merchant.get_actual_prep_time(congestion)
                    estimated_prep = actual_prep + self._rng.normal(0, 1.5)

                    # Travel time for the rider
                    distance = self._rng.exponential(sim.avg_pickup_distance_km)
                    travel_time = (distance / sim.rider_speed_kmh) * 60  # minutes

                    # FOR marking delay
                    marking_delay = merchant.get_marking_delay()

                    # Build lifecycle events
                    order_placed = OrderEvent(
                        event_type=EventType.ORDER_PLACED,
                        order_id=order_id,
                        merchant_id=merchant_id,
                        timestamp=order_time,
                        estimated_prep_time_minutes=round(estimated_prep, 2),
                        merchant_estimated_kpt_minutes=round(estimated_prep + self._rng.normal(0, 2), 2),
                    )
                    pending_events.append((current_minute, order_placed))

                    # Rider assigned ~1-3 min after order
                    rider_delay = self._rng.uniform(1, 3)
                    rider_assigned = RiderEvent(
                        event_type=EventType.RIDER_ASSIGNED,
                        rider_id=f"rider_{self._rng.integers(1000):04d}",
                        order_id=order_id,
                        merchant_id=merchant_id,
                        timestamp=order_time + timedelta(minutes=rider_delay),
                        estimated_travel_time_minutes=round(travel_time, 2),
                    )
                    pending_events.append((current_minute + rider_delay, rider_assigned))

                    # Rider arrives at merchant
                    arrival_minute = current_minute + rider_delay + travel_time
                    rider_arrived = RiderEvent(
                        event_type=EventType.RIDER_ARRIVED,
                        rider_id=rider_assigned.rider_id,
                        order_id=order_id,
                        merchant_id=merchant_id,
                        timestamp=order_time + timedelta(minutes=rider_delay + travel_time),
                        latitude=19.076 + self._rng.normal(0, 0.01),
                        longitude=72.877 + self._rng.normal(0, 0.01),
                    )
                    pending_events.append((arrival_minute, rider_arrived))

                    # FOR marking (noisy)
                    for_time = current_minute + actual_prep + marking_delay
                    for_event = FOREvent(
                        merchant_id=merchant_id,
                        order_id=order_id,
                        timestamp=order_time + timedelta(minutes=actual_prep + marking_delay),
                        marking_delay_minutes=round(marking_delay, 2),
                    )
                    pending_events.append((for_time, for_event))

                    # Order prepared (actual)
                    order_prepared = OrderEvent(
                        event_type=EventType.ORDER_PREPARED,
                        order_id=order_id,
                        merchant_id=merchant_id,
                        timestamp=order_time + timedelta(minutes=actual_prep),
                        actual_prep_time_minutes=round(actual_prep, 2),
                        estimated_prep_time_minutes=round(estimated_prep, 2),
                    )
                    pending_events.append((current_minute + actual_prep, order_prepared))

                    # Track active order
                    merchant.active_orders.append({
                        "order_id": order_id,
                        "start": current_minute,
                        "end": current_minute + actual_prep,
                    })

                # Clean up completed orders
                merchant.active_orders = [
                    o for o in merchant.active_orders
                    if o["end"] > current_minute
                ]

            # Sort pending events by time and yield those in current window
            pending_events.sort(key=lambda x: x[0])

            while pending_events and pending_events[0][0] <= current_minute + 1:
                _, event = pending_events.pop(0)

                # Inject duplicate events (~2%)
                if inject_duplicates and self._rng.random() < 0.02:
                    yield event  # Yield the original
                    yield event  # Yield duplicate (same event_id)
                    continue

                # Inject late events (~3%): shift timestamp backwards
                if inject_late_events and self._rng.random() < 0.03:
                    if isinstance(event, (OrderEvent, RiderEvent)):
                        event.timestamp = event.timestamp - timedelta(
                            minutes=self._rng.uniform(2, 5)
                        )

                yield event

            current_minute += 1.0
            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.001)

        # Flush remaining events
        pending_events.sort(key=lambda x: x[0])
        for _, event in pending_events:
            yield event
