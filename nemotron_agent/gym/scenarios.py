"""Grid topology scenarios for the gym.

Each scenario defines a complete grid: buses, branches, storage, and weather
sequences.  These range from small test cases to realistic regional grids.

Disturbance profiles inject time-varying load ramps, renewable fluctuation,
and contingencies (line trips, gen outages) into the simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from nemotron_agent.gym.power_flow import Branch, Bus, StorageUnit


@dataclass
class WeatherProfile:
    """Time-varying weather that drives renewable output and load."""

    solar_peak_w_m2: float = 900.0
    wind_mean_m_s: float = 8.0
    wind_std_m_s: float = 3.0
    temp_mean_c: float = 28.0
    temp_std_c: float = 5.0

    def sample(self, hour_of_day: float) -> dict[str, float]:
        # Solar follows a sinusoidal curve peaking at noon
        solar = max(0.0, self.solar_peak_w_m2 * math.sin(
            math.pi * max(0.0, min(hour_of_day - 6, 12)) / 12.0
        )) if 6 <= hour_of_day <= 18 else 0.0
        solar *= random.uniform(0.7, 1.0)  # cloud noise

        wind = max(0.0, random.gauss(self.wind_mean_m_s, self.wind_std_m_s))
        temp = random.gauss(self.temp_mean_c, self.temp_std_c)
        return {
            "solar_w_m2": solar,
            "wind_m_s": wind,
            "temperature_c": temp,
        }


@dataclass
class Disturbance:
    """A discrete event injected at a specific timestep."""

    timestep: int
    kind: str                    # "line_trip" | "gen_outage" | "load_spike" | "line_restore"
    target_id: str               # branch_id or bus_id
    magnitude: float = 0.0      # MW for load_spike, ignored for trips


@dataclass
class Scenario:
    """Complete grid scenario: topology + weather + disturbances."""

    name: str
    buses: list[Bus]
    branches: list[Branch]
    storage: list[StorageUnit]
    weather: WeatherProfile
    disturbances: list[Disturbance] = field(default_factory=list)
    episode_hours: float = 24.0
    steps_per_hour: int = 12     # 5-minute intervals


# ── Built-in scenarios ───────────────────────────────────

def six_bus_scenario() -> Scenario:
    """Small 6-bus test grid with renewables, storage, and congestion.

    Topology (roughly):
        [SOLAR]B0 ──LN01── B1[CITY-LOAD] ──LN12── B2[GAS-GEN]
               │                  │                     │
             LN03              LN14                  LN25
               │                  │                     │
        [WIND]B3 ──LN34── B4[SUBURB+BESS] ──LN45── B5[COAL-GEN,SLACK]

    The grid starts stressed — loads are near line limits, and immediate
    disturbances hit within the first few steps so the agent must act
    from the very beginning.
    """
    buses = [
        Bus("B0-SOLAR", 0, gen_mw=80, gen_max_mw=120, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=10),
        Bus("B1-CITY", 1, load_mw=210, flexible_load_mw=30),       # heavier city load
        Bus("B2-GAS", 2, gen_mw=100, gen_max_mw=200, gen_min_mw=20,
            gen_cost_per_mwh=55, gen_emission_tco2_per_mwh=0.4, load_mw=40),
        Bus("B3-WIND", 3, gen_mw=60, gen_max_mw=100, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=15),
        Bus("B4-SUBURB", 4, load_mw=130, flexible_load_mw=40),     # heavier suburb load
        Bus("B5-COAL", 5, is_slack=True, gen_mw=150, gen_max_mw=300, gen_min_mw=50,
            gen_cost_per_mwh=35, gen_emission_tco2_per_mwh=0.9, load_mw=30),
    ]
    branches = [
        Branch("LN01", 0, 1, x_pu=0.05, capacity_mw=90),   # tighter capacity
        Branch("LN12", 1, 2, x_pu=0.04, capacity_mw=110),  # tighter capacity
        Branch("LN03", 0, 3, x_pu=0.06, capacity_mw=80),
        Branch("LN14", 1, 4, x_pu=0.05, capacity_mw=90),
        Branch("LN25", 2, 5, x_pu=0.03, capacity_mw=150),
        Branch("LN34", 3, 4, x_pu=0.07, capacity_mw=70),
        Branch("LN45", 4, 5, x_pu=0.04, capacity_mw=105),  # tighter capacity
    ]
    storage = [
        StorageUnit("BESS-SUBURB", bus_idx=4, soc_mwh=40, capacity_mwh=100, max_power_mw=50),
    ]
    disturbances = [
        # ─── Immediate problems (agent must act from step 1) ───
        Disturbance(timestep=1, kind="load_spike", target_id="B1-CITY", magnitude=30),
        Disturbance(timestep=3, kind="line_trip", target_id="LN03"),
        # ─── Midday: line restored but new load surge ──────────
        Disturbance(timestep=48, kind="line_restore", target_id="LN03"),
        Disturbance(timestep=48, kind="line_trip", target_id="LN12"),
        Disturbance(timestep=72, kind="line_restore", target_id="LN12"),
        # ─── Evening peak: heavy city demand ───────────────────
        Disturbance(timestep=84, kind="load_spike", target_id="B1-CITY", magnitude=60),
        Disturbance(timestep=96, kind="load_spike", target_id="B4-SUBURB", magnitude=35),
    ]
    return Scenario(
        name="6-bus-renewables",
        buses=buses,
        branches=branches,
        storage=storage,
        weather=WeatherProfile(solar_peak_w_m2=900, wind_mean_m_s=8),
        disturbances=disturbances,
    )


def fourteen_bus_scenario() -> Scenario:
    """IEEE 14-bus inspired scenario with DERs and storage added.

    Standard IEEE 14-bus topology with:
      - Solar at bus 2, 6, 8
      - Wind at bus 3
      - Battery storage at bus 9, 14
      - High urban load at bus 4, 5, 10, 11
    """
    buses = [
        Bus("B01", 0, is_slack=True, gen_mw=230, gen_max_mw=400, gen_min_mw=50,
            gen_cost_per_mwh=30, gen_emission_tco2_per_mwh=0.5, load_mw=0),
        Bus("B02", 1, gen_mw=40, gen_max_mw=80, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=21.7),
        Bus("B03", 2, gen_mw=25, gen_max_mw=60, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=94.2),
        Bus("B04", 3, load_mw=47.8, flexible_load_mw=10),
        Bus("B05", 4, load_mw=7.6, flexible_load_mw=2),
        Bus("B06", 5, gen_mw=30, gen_max_mw=50, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=11.2),
        Bus("B07", 6, load_mw=0),
        Bus("B08", 7, gen_mw=20, gen_max_mw=40, gen_min_mw=0,
            gen_cost_per_mwh=0, is_renewable=True, load_mw=0),
        Bus("B09", 8, load_mw=29.5, flexible_load_mw=8),
        Bus("B10", 9, load_mw=9.0),
        Bus("B11", 10, load_mw=3.5),
        Bus("B12", 11, load_mw=6.1),
        Bus("B13", 12, load_mw=13.5),
        Bus("B14", 13, load_mw=14.9, flexible_load_mw=5),
    ]
    branches = [
        Branch("LN-01-02", 0, 1, x_pu=0.06, capacity_mw=120),
        Branch("LN-01-05", 0, 4, x_pu=0.22, capacity_mw=80),
        Branch("LN-02-03", 1, 2, x_pu=0.20, capacity_mw=70),
        Branch("LN-02-04", 1, 3, x_pu=0.18, capacity_mw=70),
        Branch("LN-02-05", 1, 4, x_pu=0.17, capacity_mw=60),
        Branch("LN-03-04", 2, 3, x_pu=0.17, capacity_mw=60),
        Branch("LN-04-05", 3, 4, x_pu=0.04, capacity_mw=90),
        Branch("LN-04-07", 3, 6, x_pu=0.21, capacity_mw=50),
        Branch("LN-04-09", 3, 8, x_pu=0.56, capacity_mw=40),
        Branch("LN-05-06", 4, 5, x_pu=0.25, capacity_mw=50),
        Branch("LN-06-11", 5, 10, x_pu=0.20, capacity_mw=30),
        Branch("LN-06-12", 5, 11, x_pu=0.26, capacity_mw=30),
        Branch("LN-06-13", 5, 12, x_pu=0.13, capacity_mw=30),
        Branch("LN-07-08", 6, 7, x_pu=0.18, capacity_mw=40),
        Branch("LN-07-09", 6, 8, x_pu=0.11, capacity_mw=40),
        Branch("LN-09-10", 8, 9, x_pu=0.08, capacity_mw=35),
        Branch("LN-09-14", 8, 13, x_pu=0.27, capacity_mw=30),
        Branch("LN-10-11", 9, 10, x_pu=0.19, capacity_mw=25),
        Branch("LN-12-13", 11, 12, x_pu=0.20, capacity_mw=25),
        Branch("LN-13-14", 12, 13, x_pu=0.35, capacity_mw=25),
    ]
    storage = [
        StorageUnit("BESS-09", bus_idx=8, soc_mwh=30, capacity_mwh=80, max_power_mw=40),
        StorageUnit("BESS-14", bus_idx=13, soc_mwh=15, capacity_mwh=40, max_power_mw=20),
    ]
    disturbances = [
        Disturbance(timestep=36, kind="line_trip", target_id="LN-01-02"),
        Disturbance(timestep=60, kind="line_restore", target_id="LN-01-02"),
        Disturbance(timestep=96, kind="load_spike", target_id="B04", magnitude=30),
        Disturbance(timestep=108, kind="gen_outage", target_id="B01"),
    ]
    return Scenario(
        name="14-bus-ieee-der",
        buses=buses,
        branches=branches,
        storage=storage,
        weather=WeatherProfile(solar_peak_w_m2=850, wind_mean_m_s=7, temp_mean_c=30),
        disturbances=disturbances,
    )


SCENARIOS: dict[str, callable] = {
    "6-bus": six_bus_scenario,
    "14-bus": fourteen_bus_scenario,
}
