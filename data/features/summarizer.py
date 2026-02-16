"""Text summarizer — converts GridState + market context into a natural
language summary that goes into Nemotron's context window.

The agent sees both the structured GridState block AND this textual summary,
giving it richer context for reasoning about grid conditions.
"""

from __future__ import annotations

import logging
from typing import Any

from nemotron_agent.grid_state import GridState

logger = logging.getLogger(__name__)


class StateSummarizer:
    """Generates human-readable grid summaries for the Nemotron context."""

    def summarize(
        self,
        state: GridState,
        market: dict[str, Any] | None = None,
        forecasts: dict[str, Any] | None = None,
    ) -> str:
        """Build a text summary from structured grid state + market data."""
        parts = []

        # Overall system status
        gen = state.total_generation_mw
        load = state.total_load_mw
        balance = gen - load
        parts.append(
            f"System is {'generation-heavy' if balance > 0 else 'load-heavy'} "
            f"with {abs(balance):.0f} MW {'surplus' if balance > 0 else 'deficit'}. "
            f"Total generation: {gen:.0f} MW, total load: {load:.0f} MW."
        )

        # Congestion
        congested = state.congested_lines
        if congested:
            lines_str = ", ".join(
                f"{l.line_id} at {l.loading_pct:.0f}%" for l in congested
            )
            parts.append(
                f"CONGESTION: {len(congested)} line(s) above 90% loading: {lines_str}. "
                f"Risk of thermal overload if not addressed."
            )
        else:
            parts.append("No transmission congestion detected.")

        # Voltage
        violations = state.voltage_violations
        if violations:
            buses_str = ", ".join(
                f"{b.bus_id} ({b.voltage_pu:.3f} pu)" for b in violations
            )
            parts.append(
                f"VOLTAGE: {len(violations)} bus(es) outside limits: {buses_str}."
            )

        # Storage
        if state.storage:
            for s in state.storage:
                status = "idle"
                if s.current_setpoint_mw > 0:
                    status = f"discharging {s.current_setpoint_mw:.0f} MW"
                elif s.current_setpoint_mw < 0:
                    status = f"charging {abs(s.current_setpoint_mw):.0f} MW"
                parts.append(
                    f"Storage {s.unit_id}: SoC {s.soc_pct:.0f}%, {status}."
                )

        # Weather
        wx = state.weather
        parts.append(
            f"Weather: solar {wx.solar_irradiance_w_m2:.0f} W/m², "
            f"wind {wx.wind_speed_m_s:.1f} m/s, "
            f"temp {wx.temperature_c:.0f}°C, "
            f"clouds {wx.cloud_cover_pct:.0f}%."
        )

        # Market context
        if market:
            lmps = market.get("lmps", {})
            if lmps:
                avg_lmp = sum(lmps.values()) / len(lmps)
                max_node = max(lmps, key=lmps.get) if lmps else "N/A"
                parts.append(
                    f"Market: avg LMP ${avg_lmp:.1f}/MWh, "
                    f"highest at {max_node} (${lmps.get(max_node, 0):.1f}/MWh)."
                )
            carbon = market.get("carbon_price", 0)
            if carbon:
                parts.append(f"Carbon price: ${carbon:.1f}/ton CO₂.")

        # Forecasts
        if forecasts:
            solar_fcst = forecasts.get("solar_w_m2", [])
            wind_fcst = forecasts.get("wind_m_s", [])
            if solar_fcst:
                peak_solar = max(solar_fcst)
                peak_hour = solar_fcst.index(peak_solar)
                parts.append(
                    f"Solar forecast: peak {peak_solar:.0f} W/m² in {peak_hour}h."
                )
            if wind_fcst:
                avg_wind = sum(wind_fcst) / len(wind_fcst)
                parts.append(f"Wind forecast: avg {avg_wind:.1f} m/s over next {len(wind_fcst)}h.")

        # Alerts
        if state.alerts:
            parts.append("Active alerts: " + " | ".join(state.alerts))

        return "\n".join(parts)
