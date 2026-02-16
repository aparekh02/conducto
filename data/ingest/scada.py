"""SCADA telemetry ingestion.

Supervisory Control and Data Acquisition — the primary source of
real-time grid measurements:
  - Bus voltages (kV)
  - Line flows (MW, MVAr)
  - Transformer tap positions
  - Breaker/switch status (open/closed)
  - Generator output (MW)

Publishes to topic: grid.scada
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from data.bus import Message, MessageBus, Topic

logger = logging.getLogger(__name__)


class SCADAIngest:
    """Ingests SCADA telemetry and publishes to the message bus."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus

    def publish_bus_reading(
        self,
        bus_id: str,
        voltage_kv: float,
        load_mw: float,
        gen_mw: float,
        frequency_hz: float = 60.0,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.SCADA,
            key=f"bus:{bus_id}",
            value={
                "type": "bus",
                "bus_id": bus_id,
                "voltage_kv": voltage_kv,
                "load_mw": load_mw,
                "gen_mw": gen_mw,
                "frequency_hz": frequency_hz,
            },
        ))

    def publish_line_reading(
        self,
        line_id: str,
        from_bus: str,
        to_bus: str,
        flow_mw: float,
        flow_mvar: float,
        capacity_mw: float,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.SCADA,
            key=f"line:{line_id}",
            value={
                "type": "line",
                "line_id": line_id,
                "from_bus": from_bus,
                "to_bus": to_bus,
                "flow_mw": flow_mw,
                "flow_mvar": flow_mvar,
                "capacity_mw": capacity_mw,
                "loading_pct": abs(flow_mw) / capacity_mw * 100 if capacity_mw > 0 else 0,
            },
        ))

    def publish_switch_status(
        self, switch_id: str, is_closed: bool,
    ) -> None:
        self.bus.publish(Message(
            topic=Topic.SCADA,
            key=f"switch:{switch_id}",
            value={
                "type": "switch",
                "switch_id": switch_id,
                "is_closed": is_closed,
            },
        ))

    # ── Demo data generator ──────────────────────────────

    def generate_demo_snapshot(self, bus_ids: list[str], line_specs: list[dict]) -> None:
        """Publish a realistic demo snapshot for testing."""
        for bid in bus_ids:
            self.publish_bus_reading(
                bus_id=bid,
                voltage_kv=230 * random.uniform(0.94, 1.06),
                load_mw=random.uniform(20, 250),
                gen_mw=random.uniform(0, 200),
                frequency_hz=60.0 + random.gauss(0, 0.02),
            )
        for spec in line_specs:
            cap = spec["capacity_mw"]
            flow = random.uniform(-cap * 0.3, cap * 1.1)
            self.publish_line_reading(
                line_id=spec["line_id"],
                from_bus=spec["from_bus"],
                to_bus=spec["to_bus"],
                flow_mw=flow,
                flow_mvar=flow * random.uniform(0.1, 0.3),
                capacity_mw=cap,
            )
