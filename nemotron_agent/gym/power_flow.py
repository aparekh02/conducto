"""DC power flow solver — the physics engine inside the grid gym.

Uses the linearised DC power flow approximation:
    P = B * θ
where B is the bus susceptance matrix and θ is the vector of voltage angles.
Line flows are derived from angle differences across each branch.

This is fast enough to run thousands of steps per second on CPU, which is
what we need for the RL training loop.  It captures the essential congestion
and dispatch physics without the complexity of full AC power flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bus:
    bus_id: str
    index: int                      # position in the B-matrix
    is_slack: bool = False          # one bus must be the reference
    v_pu: float = 1.0              # voltage magnitude (updated heuristically)
    load_mw: float = 0.0
    gen_mw: float = 0.0
    gen_max_mw: float = 0.0
    gen_min_mw: float = 0.0
    gen_cost_per_mwh: float = 30.0  # $/MWh marginal cost
    gen_emission_tco2_per_mwh: float = 0.0
    is_renewable: bool = False
    flexible_load_mw: float = 0.0   # portion of load that can be shaped


@dataclass
class Branch:
    branch_id: str
    from_idx: int
    to_idx: int
    x_pu: float                     # reactance in per-unit
    capacity_mw: float = 100.0
    is_closed: bool = True

    @property
    def b_pu(self) -> float:
        """Susceptance = 1 / reactance."""
        return 1.0 / self.x_pu if self.x_pu != 0 else 1e6


@dataclass
class StorageUnit:
    unit_id: str
    bus_idx: int
    soc_mwh: float = 0.0
    capacity_mwh: float = 100.0
    max_power_mw: float = 50.0
    efficiency: float = 0.92


@dataclass
class PowerFlowResult:
    """Output of a single DC power flow solve."""

    angles_rad: np.ndarray
    bus_injections_mw: np.ndarray
    line_flows_mw: np.ndarray
    line_loading_pct: np.ndarray
    converged: bool = True


class PowerFlowSolver:
    """Stateless DC power flow solver operating on grid topology arrays."""

    def __init__(
        self,
        buses: list[Bus],
        branches: list[Branch],
        base_mva: float = 100.0,
    ) -> None:
        self.buses = buses
        self.branches = branches
        self.base_mva = base_mva
        self.n_bus = len(buses)
        self.n_branch = len(branches)
        self.slack_idx = next(b.index for b in buses if b.is_slack)
        self._build_b_matrix()

    def _build_b_matrix(self) -> None:
        """Build the bus susceptance matrix from branch data."""
        B = np.zeros((self.n_bus, self.n_bus))
        for br in self.branches:
            if not br.is_closed:
                continue
            b = br.b_pu
            B[br.from_idx, br.to_idx] -= b
            B[br.to_idx, br.from_idx] -= b
            B[br.from_idx, br.from_idx] += b
            B[br.to_idx, br.to_idx] += b
        self._B_full = B

    def rebuild(self) -> None:
        """Rebuild B-matrix after topology changes (open/close branches)."""
        self._build_b_matrix()

    def solve(self) -> PowerFlowResult:
        """Solve DC power flow → angles, line flows, loading percentages."""
        # Net injection at each bus (gen - load) in per-unit
        p_inj = np.array([
            (b.gen_mw - b.load_mw) / self.base_mva for b in self.buses
        ])

        # Remove slack row/col, solve B_red * θ_red = P_red
        mask = np.ones(self.n_bus, dtype=bool)
        mask[self.slack_idx] = False
        B_red = self._B_full[np.ix_(mask, mask)]
        p_red = p_inj[mask]

        try:
            theta_red = np.linalg.solve(B_red, p_red)
        except np.linalg.LinAlgError:
            # Singular matrix — grid is islanded or degenerate
            return PowerFlowResult(
                angles_rad=np.zeros(self.n_bus),
                bus_injections_mw=p_inj * self.base_mva,
                line_flows_mw=np.zeros(self.n_branch),
                line_loading_pct=np.zeros(self.n_branch),
                converged=False,
            )

        # Re-insert slack angle = 0
        theta = np.zeros(self.n_bus)
        theta[mask] = theta_red

        # Line flows
        flows = np.zeros(self.n_branch)
        for i, br in enumerate(self.branches):
            if br.is_closed:
                flows[i] = br.b_pu * (theta[br.from_idx] - theta[br.to_idx]) * self.base_mva
            else:
                flows[i] = 0.0

        loading = np.array([
            abs(flows[i]) / br.capacity_mw * 100.0 if br.capacity_mw > 0 else 0.0
            for i, br in enumerate(self.branches)
        ])

        # Heuristic voltage update (DC PF doesn't solve voltage, but we
        # approximate based on angle spread and reactive heuristics)
        max_angle_spread = np.ptp(theta) if len(theta) > 1 else 0.0
        for bus in self.buses:
            # Voltage drops roughly with angle deviation and loading
            bus.v_pu = 1.0 - 0.02 * abs(theta[bus.index]) - 0.01 * (
                bus.load_mw / max(bus.gen_mw, 1.0) - 1.0
            ) * 0.05
            bus.v_pu = np.clip(bus.v_pu, 0.85, 1.10)

        return PowerFlowResult(
            angles_rad=theta,
            bus_injections_mw=p_inj * self.base_mva,
            line_flows_mw=flows,
            line_loading_pct=loading,
            converged=True,
        )
