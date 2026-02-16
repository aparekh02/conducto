"""Feature service — converts raw bus messages into structured GridState.

Subscribes to all ingest topics, maintains a rolling window of the most
recent readings per sensor, and produces a GridState snapshot on demand
that the Nemotron agent can consume.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from data.bus import Message, MessageBus, Topic
from nemotron_agent.grid_state import (
    BusReading,
    GridState,
    LineReading,
    StorageUnit,
    WeatherSnapshot,
)

logger = logging.getLogger(__name__)


class StateBuilder:
    """Builds GridState from streaming telemetry on the message bus."""

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        # Latest reading per key, per topic
        self._latest: dict[str, dict[str, dict[str, Any]]] = {
            Topic.SCADA: {},
            Topic.PMU: {},
            Topic.WEATHER: {},
            Topic.MARKET: {},
        }
        self._alerts: list[str] = []

        # Subscribe to all topics
        for topic in (Topic.SCADA, Topic.PMU, Topic.WEATHER, Topic.MARKET, Topic.ALERTS):
            bus.subscribe(topic, self._on_message)

        logger.info("StateBuilder subscribed to all ingest topics")

    def _on_message(self, msg: Message) -> None:
        """Update the latest reading for this key."""
        if msg.topic == Topic.ALERTS:
            self._alerts.append(msg.value.get("text", str(msg.value)))
            self._alerts = self._alerts[-20:]  # keep last 20
        elif msg.topic in self._latest:
            self._latest[msg.topic][msg.key] = msg.value

    # ── Build GridState ──────────────────────────────────

    def build(self) -> GridState:
        """Assemble a GridState from the latest readings across all topics."""
        buses = self._build_buses()
        lines = self._build_lines()
        weather = self._build_weather()
        storage = self._build_storage()

        return GridState(
            timestamp=time.time(),
            buses=buses,
            lines=lines,
            storage=storage,
            weather=weather,
            alerts=list(self._alerts),
        )

    def _build_buses(self) -> list[BusReading]:
        buses = []
        for key, val in self._latest[Topic.SCADA].items():
            if not key.startswith("bus:"):
                continue
            # Enrich with PMU data if available
            pmu_key = f"pmu:{val['bus_id']}"
            pmu = self._latest[Topic.PMU].get(pmu_key, {})
            freq = pmu.get("frequency_hz", val.get("frequency_hz", 60.0))

            base_kv = 230.0  # nominal
            voltage_pu = val["voltage_kv"] / base_kv if "voltage_kv" in val else 1.0

            buses.append(BusReading(
                bus_id=val["bus_id"],
                voltage_pu=voltage_pu,
                load_mw=val.get("load_mw", 0),
                generation_mw=val.get("gen_mw", 0),
                frequency_hz=freq,
            ))
        return buses

    def _build_lines(self) -> list[LineReading]:
        lines = []
        for key, val in self._latest[Topic.SCADA].items():
            if not key.startswith("line:"):
                continue
            lines.append(LineReading(
                line_id=val["line_id"],
                from_bus=val["from_bus"],
                to_bus=val["to_bus"],
                flow_mw=val["flow_mw"],
                capacity_mw=val["capacity_mw"],
                loading_pct=val.get("loading_pct", 0),
            ))
        return lines

    def _build_weather(self) -> WeatherSnapshot:
        # Average across all weather stations
        wx_readings = [
            v for k, v in self._latest[Topic.WEATHER].items()
            if not k.startswith("wx_fcst:")
        ]
        if not wx_readings:
            return WeatherSnapshot()

        n = len(wx_readings)
        return WeatherSnapshot(
            solar_irradiance_w_m2=sum(w.get("solar_irradiance_w_m2", 0) for w in wx_readings) / n,
            wind_speed_m_s=sum(w.get("wind_speed_m_s", 0) for w in wx_readings) / n,
            temperature_c=sum(w.get("temperature_c", 25) for w in wx_readings) / n,
            cloud_cover_pct=sum(w.get("cloud_cover_pct", 0) for w in wx_readings) / n,
        )

    def _build_storage(self) -> list[StorageUnit]:
        # Storage comes through SCADA as well
        storage = []
        for key, val in self._latest[Topic.SCADA].items():
            if val.get("type") != "storage":
                continue
            storage.append(StorageUnit(
                unit_id=val["unit_id"],
                soc_pct=val.get("soc_pct", 50),
                max_power_mw=val.get("max_power_mw", 50),
                current_setpoint_mw=val.get("setpoint_mw", 0),
            ))
        return storage

    # ── Market context (for cost-aware decisions) ────────

    def get_market_context(self) -> dict[str, Any]:
        """Return the latest market signals as a dict for the agent prompt."""
        lmps = {}
        ancillary = {}
        carbon = 0.0
        for key, val in self._latest[Topic.MARKET].items():
            if val.get("type") == "lmp":
                lmps[val["node_id"]] = val["lmp_total"]
            elif val.get("type") == "ancillary":
                ancillary = val
            elif val.get("type") == "carbon":
                carbon = val["price_usd_per_ton"]
        return {"lmps": lmps, "ancillary": ancillary, "carbon_price": carbon}
