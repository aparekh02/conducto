"""Phasor Measurement Unit (PMU) ingestion.

PMUs provide high-frequency (30–60 samples/sec) synchronised phasor
measurements — the fastest view into grid dynamics:
  - Voltage phasors (magnitude + angle)
  - Current phasors
  - Frequency and rate-of-change-of-frequency (ROCOF)
  - Power (real + reactive)

Critical for detecting oscillations, islanding, and transient instability
that SCADA (2–4 sec resolution) would miss.

Publishes to topic: grid.pmu
"""

from __future__ import annotations

import logging
import math
import random
import time

from data.bus import Message, MessageBus, Topic

logger = logging.getLogger(__name__)


class PMUIngest:
    """Ingests PMU phasor data and publishes to the message bus."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus

    def publish_phasor(
        self,
        pmu_id: str,
        bus_id: str,
        voltage_mag_kv: float,
        voltage_angle_deg: float,
        current_mag_a: float,
        current_angle_deg: float,
        frequency_hz: float,
        rocof_hz_per_s: float = 0.0,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.PMU,
            key=f"pmu:{pmu_id}",
            value={
                "pmu_id": pmu_id,
                "bus_id": bus_id,
                "voltage_mag_kv": voltage_mag_kv,
                "voltage_angle_deg": voltage_angle_deg,
                "current_mag_a": current_mag_a,
                "current_angle_deg": current_angle_deg,
                "frequency_hz": frequency_hz,
                "rocof_hz_per_s": rocof_hz_per_s,
                "power_mw": (
                    voltage_mag_kv * current_mag_a * math.sqrt(3)
                    * math.cos(math.radians(voltage_angle_deg - current_angle_deg))
                    / 1000
                ),
                "power_mvar": (
                    voltage_mag_kv * current_mag_a * math.sqrt(3)
                    * math.sin(math.radians(voltage_angle_deg - current_angle_deg))
                    / 1000
                ),
            },
        ))

    def generate_demo_stream(self, pmu_specs: list[dict], n_samples: int = 30) -> None:
        """Generate a burst of PMU samples at ~30 Hz for testing."""
        for _ in range(n_samples):
            for spec in pmu_specs:
                self.publish_phasor(
                    pmu_id=spec["pmu_id"],
                    bus_id=spec["bus_id"],
                    voltage_mag_kv=230 * random.uniform(0.95, 1.05),
                    voltage_angle_deg=random.uniform(-30, 30),
                    current_mag_a=random.uniform(50, 500),
                    current_angle_deg=random.uniform(-40, 20),
                    frequency_hz=60.0 + random.gauss(0, 0.01),
                    rocof_hz_per_s=random.gauss(0, 0.1),
                )
