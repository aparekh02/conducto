"""Grid state context — the observation window the agent reasons over.

Holds the structured snapshot of current grid conditions that gets
serialised into the Nemotron context window each decision cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusReading:
    bus_id: str
    voltage_pu: float          # per-unit voltage
    load_mw: float
    generation_mw: float
    frequency_hz: float = 60.0


@dataclass
class LineReading:
    line_id: str
    from_bus: str
    to_bus: str
    flow_mw: float
    capacity_mw: float
    loading_pct: float         # flow / capacity * 100


@dataclass
class StorageUnit:
    unit_id: str
    soc_pct: float             # state of charge 0-100
    max_power_mw: float
    current_setpoint_mw: float # positive = discharging


@dataclass
class WeatherSnapshot:
    solar_irradiance_w_m2: float = 0.0
    wind_speed_m_s: float = 0.0
    temperature_c: float = 25.0
    cloud_cover_pct: float = 0.0


@dataclass
class GridState:
    """Complete observation at a single timestep."""

    timestamp: float = field(default_factory=time.time)
    buses: list[BusReading] = field(default_factory=list)
    lines: list[LineReading] = field(default_factory=list)
    storage: list[StorageUnit] = field(default_factory=list)
    weather: WeatherSnapshot = field(default_factory=WeatherSnapshot)
    alerts: list[str] = field(default_factory=list)

    # ── Derived metrics ──────────────────────────────────

    @property
    def congested_lines(self) -> list[LineReading]:
        return [l for l in self.lines if l.loading_pct > 90]

    @property
    def voltage_violations(self) -> list[BusReading]:
        return [b for b in self.buses if not (0.95 <= b.voltage_pu <= 1.05)]

    @property
    def total_load_mw(self) -> float:
        return sum(b.load_mw for b in self.buses)

    @property
    def total_generation_mw(self) -> float:
        return sum(b.generation_mw for b in self.buses)

    # ── Serialisation for the Nemotron context ───────────

    def to_context_block(self) -> str:
        """Render this state as a compact text block for the LLM prompt."""
        parts = [f"[GRID STATE  t={self.timestamp:.0f}]"]

        parts.append(
            f"TOTALS  load={self.total_load_mw:.1f} MW  "
            f"gen={self.total_generation_mw:.1f} MW  "
            f"balance={self.total_generation_mw - self.total_load_mw:+.1f} MW"
        )

        # Show ALL lines so the model can see the full network state
        if self.lines:
            parts.append("LINES:")
            for l in self.lines:
                tag = " ** CONGESTED **" if l.loading_pct > 90 else ""
                parts.append(
                    f"  {l.line_id} ({l.from_bus}→{l.to_bus})  "
                    f"flow={l.flow_mw:.1f}/{l.capacity_mw:.0f}MW  "
                    f"loading={l.loading_pct:.0f}%{tag}"
                )

        if self.voltage_violations:
            parts.append("VOLTAGE VIOLATIONS:")
            for b in self.voltage_violations:
                parts.append(f"  {b.bus_id}  V={b.voltage_pu:.4f} pu")

        if self.storage:
            parts.append("STORAGE:")
            for s in self.storage:
                parts.append(
                    f"  {s.unit_id}  SoC={s.soc_pct:.0f}%  "
                    f"setpoint={s.current_setpoint_mw:+.1f} MW"
                )

        parts.append(
            f"WEATHER  solar={self.weather.solar_irradiance_w_m2:.0f} W/m²  "
            f"wind={self.weather.wind_speed_m_s:.1f} m/s  "
            f"temp={self.weather.temperature_c:.1f}°C  "
            f"clouds={self.weather.cloud_cover_pct:.0f}%"
        )

        if self.alerts:
            parts.append("ALERTS: " + " | ".join(self.alerts))

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for logging / replay buffer)."""
        import dataclasses

        def _ser(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _ser(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_ser(i) for i in obj]
            return obj

        return _ser(self)
