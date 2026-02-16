"""Multi-objective reward function for the grid gym.

Penalises:
  - Line overloads (congestion)
  - Voltage violations (outside 0.95–1.05 pu)
  - Renewable curtailment
  - Grid instability (generation-load imbalance)
  - Operating cost
  - Carbon emissions

Rewards:
  - Reliability (all loads served)
  - Low cost
  - Low emissions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nemotron_agent.gym.power_flow import Bus, PowerFlowResult


@dataclass
class RewardWeights:
    """Tunable weights for each objective.  All positive; penalties are negated."""

    congestion: float = 5.0
    voltage: float = 3.0
    curtailment: float = 2.0
    imbalance: float = 4.0
    cost: float = 0.5
    emissions: float = 1.0
    reliability: float = 10.0


@dataclass
class RewardBreakdown:
    """Itemised reward so we can log what's driving the score."""

    congestion_penalty: float = 0.0
    voltage_penalty: float = 0.0
    curtailment_penalty: float = 0.0
    imbalance_penalty: float = 0.0
    cost_penalty: float = 0.0
    emissions_penalty: float = 0.0
    reliability_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "congestion": self.congestion_penalty,
            "voltage": self.voltage_penalty,
            "curtailment": self.curtailment_penalty,
            "imbalance": self.imbalance_penalty,
            "cost": self.cost_penalty,
            "emissions": self.emissions_penalty,
            "reliability": self.reliability_bonus,
            "total": self.total,
        }


def compute_reward(
    pf: PowerFlowResult,
    buses: list[Bus],
    curtailed_mw: float,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Compute the scalar reward from a power-flow solution."""
    w = weights or RewardWeights()
    r = RewardBreakdown()

    # ── Congestion: penalise lines loaded above 100% ─────
    overloads = np.maximum(pf.line_loading_pct - 100.0, 0.0)
    r.congestion_penalty = -w.congestion * float(np.sum(overloads ** 2)) / 1000.0

    # ── Voltage: penalise buses outside [0.95, 1.05] pu ──
    for bus in buses:
        if bus.v_pu < 0.95:
            r.voltage_penalty -= w.voltage * (0.95 - bus.v_pu) ** 2 * 100
        elif bus.v_pu > 1.05:
            r.voltage_penalty -= w.voltage * (bus.v_pu - 1.05) ** 2 * 100

    # ── Curtailment: penalise wasted renewable energy ────
    r.curtailment_penalty = -w.curtailment * curtailed_mw / 10.0

    # ── Imbalance: gen vs load mismatch ──────────────────
    total_gen = sum(b.gen_mw for b in buses)
    total_load = sum(b.load_mw for b in buses)
    imbalance_mw = abs(total_gen - total_load)
    r.imbalance_penalty = -w.imbalance * (imbalance_mw / max(total_load, 1.0)) ** 2

    # ── Cost: total generation cost ──────────────────────
    total_cost = sum(b.gen_mw * b.gen_cost_per_mwh for b in buses if b.gen_mw > 0)
    r.cost_penalty = -w.cost * total_cost / 10000.0

    # ── Emissions ────────────────────────────────────────
    total_emissions = sum(
        b.gen_mw * b.gen_emission_tco2_per_mwh for b in buses if b.gen_mw > 0
    )
    r.emissions_penalty = -w.emissions * total_emissions / 100.0

    # ── Reliability bonus: all load served, no islands ───
    if pf.converged and imbalance_mw < total_load * 0.05:
        r.reliability_bonus = w.reliability
    elif pf.converged:
        r.reliability_bonus = w.reliability * 0.5

    r.total = (
        r.congestion_penalty
        + r.voltage_penalty
        + r.curtailment_penalty
        + r.imbalance_penalty
        + r.cost_penalty
        + r.emissions_penalty
        + r.reliability_bonus
    )
    return r
