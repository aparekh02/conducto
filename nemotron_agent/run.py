#!/usr/bin/env python3
"""Entry point — run the Nemotron Grid Agent against demo data.

Usage:
    python -m nemotron_agent.run            # single decision on demo state
    python -m nemotron_agent.run --loop     # continuous loop (Ctrl-C to stop)
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from nemotron_agent.agent import NemotronGridAgent
from nemotron_agent.config import Config
from nemotron_agent.grid_state import (
    BusReading,
    GridState,
    LineReading,
    StorageUnit,
    WeatherSnapshot,
)


def build_demo_state() -> GridState:
    """Construct a realistic-looking grid snapshot for testing."""
    return GridState(
        buses=[
            BusReading("BUS-001", voltage_pu=1.02, load_mw=120, generation_mw=130),
            BusReading("BUS-002", voltage_pu=0.94, load_mw=200, generation_mw=50),
            BusReading("BUS-003", voltage_pu=1.01, load_mw=80, generation_mw=180),
            BusReading("BUS-004", voltage_pu=1.00, load_mw=150, generation_mw=0),
        ],
        lines=[
            LineReading("LN-01", "BUS-001", "BUS-002", flow_mw=95, capacity_mw=100, loading_pct=95.0),
            LineReading("LN-02", "BUS-003", "BUS-004", flow_mw=70, capacity_mw=120, loading_pct=58.3),
            LineReading("LN-03", "BUS-001", "BUS-003", flow_mw=40, capacity_mw=80, loading_pct=50.0),
            LineReading("LN-04", "BUS-002", "BUS-004", flow_mw=88, capacity_mw=90, loading_pct=97.8),
        ],
        storage=[
            StorageUnit("BESS-A", soc_pct=72, max_power_mw=50, current_setpoint_mw=0),
            StorageUnit("BESS-B", soc_pct=30, max_power_mw=25, current_setpoint_mw=-10),
        ],
        weather=WeatherSnapshot(
            solar_irradiance_w_m2=650,
            wind_speed_m_s=8.2,
            temperature_c=31.0,
            cloud_cover_pct=40,
        ),
        alerts=[
            "LN-01 approaching thermal limit",
            "BUS-002 voltage below 0.95 pu",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nemotron Grid Pressure Autopilot")
    parser.add_argument("--loop", action="store_true", help="Run continuous decision loop")
    args = parser.parse_args()

    cfg = Config()
    logging.basicConfig(
        level=getattr(logging, cfg.agent.log_level),
        format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = NemotronGridAgent(cfg)

    if args.loop:
        print(f"Starting continuous loop (interval={cfg.agent.loop_interval_sec}s) …")
        try:
            while True:
                state = build_demo_state()
                decision = agent.decide(state)
                print(json.dumps({
                    "assessment": decision.assessment,
                    "risk_level": decision.risk_level,
                    "actions": decision.actions,
                    "latency_ms": round(decision.latency_ms),
                }, indent=2))
                print("─" * 60)
                time.sleep(cfg.agent.loop_interval_sec)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        state = build_demo_state()
        print("Grid state:\n" + state.to_context_block())
        print("─" * 60)
        decision = agent.decide(state)
        print(json.dumps({
            "assessment": decision.assessment,
            "risk_level": decision.risk_level,
            "actions": decision.actions,
            "latency_ms": round(decision.latency_ms),
        }, indent=2))


if __name__ == "__main__":
    main()
