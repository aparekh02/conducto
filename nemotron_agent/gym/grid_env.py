"""NeMo Gym-style grid environment.

Gymnasium-compatible environment that wraps the DC power flow solver,
reward function, scenarios, and disturbance injection into a standard
reset/step loop the agent (or any RL trainer) can interact with.

Observation space:
    Bus voltages, line flows & loading, generator states, storage SoC,
    weather, time-of-day, and active alerts — all flattened into a vector
    plus a human-readable text block for the LLM agent.

Action space (dict):
    redispatch  — per-generator MW delta       (continuous)
    storage     — per-storage MW setpoint       (continuous, +discharge/-charge)
    load_shape  — per-flexible-bus MW delta      (continuous)
    topology    — per-branch open/close          (discrete 0/1)
"""

from __future__ import annotations

import copy
import logging
import math
import random as _random
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nemotron_agent.gym.power_flow import Branch, Bus, PowerFlowSolver
from nemotron_agent.gym.reward import RewardBreakdown, RewardWeights, compute_reward
from nemotron_agent.gym.scenarios import Disturbance, Scenario, six_bus_scenario
from nemotron_agent.grid_state import (
    BusReading,
    GridState,
    LineReading,
    StorageUnit as GSStorageUnit,
    WeatherSnapshot,
)

logger = logging.getLogger(__name__)


class GridEnv(gym.Env):
    """NeMo Gym-style grid pressure environment."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        scenario: Scenario | None = None,
        reward_weights: RewardWeights | None = None,
        render_mode: str | None = "text",
        enable_growth: bool = False,
    ) -> None:
        super().__init__()
        self._base_scenario = scenario or six_bus_scenario()
        self.reward_weights = reward_weights or RewardWeights()
        self.render_mode = render_mode
        self.enable_growth = enable_growth

        n_bus = len(self._base_scenario.buses)
        n_branch = len(self._base_scenario.branches)
        n_gen = sum(1 for b in self._base_scenario.buses if b.gen_max_mw > 0)
        n_storage = len(self._base_scenario.storage)
        n_flex = sum(1 for b in self._base_scenario.buses if b.flexible_load_mw > 0)

        # ── Action space ─────────────────────────────────
        self.action_space = spaces.Dict({
            "redispatch": spaces.Box(-1.0, 1.0, shape=(n_gen,), dtype=np.float32),
            "storage": spaces.Box(-1.0, 1.0, shape=(max(n_storage, 1),), dtype=np.float32),
            "load_shape": spaces.Box(-1.0, 1.0, shape=(max(n_flex, 1),), dtype=np.float32),
            "topology": spaces.MultiBinary(n_branch),
        })

        # ── Observation space (flat vector) ──────────────
        # per bus: v_pu, load_mw, gen_mw (3 * n_bus)
        # per branch: flow_mw, loading_pct (2 * n_branch)
        # per storage: soc_pct, setpoint (2 * n_storage)
        # weather: solar, wind, temp (3)
        # time: hour_sin, hour_cos (2)
        obs_dim = 3 * n_bus + 2 * n_branch + 2 * max(n_storage, 1) + 3 + 2
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Indices for action mapping
        self._gen_indices = [b.index for b in self._base_scenario.buses if b.gen_max_mw > 0]
        self._flex_indices = [b.index for b in self._base_scenario.buses if b.flexible_load_mw > 0]

        # Runtime state (set in reset)
        self._scenario: Scenario | None = None
        self._solver: PowerFlowSolver | None = None
        self._step_count = 0
        self._hour = 0.0
        self._curtailed_mw = 0.0
        self._last_reward: RewardBreakdown | None = None
        self._prev_storage_inj: dict[int, float] = {}  # track previous storage injection

        # Random disturbance injection — when grid is stable, create new problems
        self._stable_steps = 0
        self._stable_threshold = 10      # ~50 min sim time before new disturbance
        self.active_events: list[dict] = []  # recent disturbance events for frontend
        self.chat_events: list[dict] = []    # chat messages for frontend

        # Growth mode state
        self._growth_timer = 0
        self._growth_interval = 7         # steps between growth events (~10s at 1.5s interval)
        self._next_bus_idx = 6            # next bus index for expansion
        self._expansion_count = 0

    # ── Gym interface ────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Deep-copy so episodes are independent
        self._scenario = copy.deepcopy(self._base_scenario)
        if self.enable_growth:
            self._scenario.disturbances = []  # growth mode generates its own events
        self._solver = PowerFlowSolver(
            self._scenario.buses, self._scenario.branches
        )
        self._step_count = 0
        self._hour = 0.0
        self._curtailed_mw = 0.0
        self._stable_steps = 0
        self._prev_storage_inj = {}
        self.active_events = []
        self.chat_events = []
        self._growth_timer = 0
        self._next_bus_idx = len(self._scenario.buses)
        self._expansion_count = 0

        for s in self._scenario.storage:
            s.soc_mwh = s.capacity_mwh * 0.5

        pf = self._solver.solve()
        obs = self._build_obs(pf)
        info = self._build_info(pf)
        return obs, info

    def step(
        self, action: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._scenario is not None, "Call reset() first"

        self._step_count += 1
        dt_hours = 1.0 / self._scenario.steps_per_hour
        self._hour += dt_hours

        # 1) Apply disturbances for this timestep
        self._apply_disturbances()

        # 2) Update weather → renewable output
        self._update_weather()

        # 3) Apply agent actions
        self._apply_actions(action)

        # 4) Run power flow
        pf = self._solver.solve()

        # 5) Compute reward
        reward_bd = compute_reward(
            pf, self._scenario.buses, self._curtailed_mw, self.reward_weights
        )
        self._last_reward = reward_bd

        # 5b) Random disturbance injection when grid is stable
        self._maybe_inject_disturbance(pf)

        # 5c) Growth events (timed, not stability-based)
        if self.enable_growth:
            n_buses_before = len(self._scenario.buses)
            n_branches_before = len(self._scenario.branches)
            self._maybe_growth_event()
            # Re-solve if topology changed so pf dimensions match
            if (len(self._scenario.buses) != n_buses_before
                    or len(self._scenario.branches) != n_branches_before):
                pf = self._solver.solve()

        # 6) Check termination
        total_steps = int(self._scenario.episode_hours * self._scenario.steps_per_hour)
        terminated = self._step_count >= total_steps
        truncated = not pf.converged

        obs = self._build_obs(pf)
        info = self._build_info(pf)
        info["reward_breakdown"] = reward_bd.to_dict()

        return obs, reward_bd.total, terminated, truncated, info

    # ── Build GridState (for the LLM agent) ──────────────

    def get_grid_state(self) -> GridState:
        """Return the current sim state as a GridState the LLM agent expects."""
        assert self._scenario is not None
        pf = self._solver.solve()
        weather = self._scenario.weather.sample(self._hour % 24)

        return GridState(
            buses=[
                BusReading(
                    bus_id=b.bus_id,
                    voltage_pu=b.v_pu,
                    load_mw=b.load_mw,
                    generation_mw=b.gen_mw,
                )
                for b in self._scenario.buses
            ],
            lines=[
                LineReading(
                    line_id=br.branch_id,
                    from_bus=self._scenario.buses[br.from_idx].bus_id,
                    to_bus=self._scenario.buses[br.to_idx].bus_id,
                    flow_mw=float(pf.line_flows_mw[i]),
                    capacity_mw=br.capacity_mw,
                    loading_pct=float(pf.line_loading_pct[i]),
                )
                for i, br in enumerate(self._scenario.branches)
            ],
            storage=[
                GSStorageUnit(
                    unit_id=s.unit_id,
                    soc_pct=s.soc_mwh / s.capacity_mwh * 100,
                    max_power_mw=s.max_power_mw,
                    current_setpoint_mw=0.0,
                )
                for s in self._scenario.storage
            ],
            weather=WeatherSnapshot(
                solar_irradiance_w_m2=weather["solar_w_m2"],
                wind_speed_m_s=weather["wind_m_s"],
                temperature_c=weather["temperature_c"],
            ),
            alerts=self._collect_alerts(pf),
        )

    # ── Internal ─────────────────────────────────────────

    def _apply_disturbances(self) -> None:
        """Inject scheduled disturbances at the current timestep."""
        for d in self._scenario.disturbances:
            if d.timestep != self._step_count:
                continue
            if d.kind == "line_trip":
                for br in self._scenario.branches:
                    if br.branch_id == d.target_id:
                        br.is_closed = False
                        self._solver.rebuild()
                        logger.info("DISTURBANCE: line %s tripped", d.target_id)
            elif d.kind == "line_restore":
                for br in self._scenario.branches:
                    if br.branch_id == d.target_id:
                        br.is_closed = True
                        self._solver.rebuild()
                        logger.info("DISTURBANCE: line %s restored", d.target_id)
            elif d.kind == "load_spike":
                for bus in self._scenario.buses:
                    if bus.bus_id == d.target_id:
                        bus.load_mw += d.magnitude
                        logger.info("DISTURBANCE: %s load +%.0f MW", d.target_id, d.magnitude)
            elif d.kind == "gen_outage":
                for bus in self._scenario.buses:
                    if bus.bus_id == d.target_id:
                        bus.gen_mw = 0
                        bus.gen_max_mw = 0
                        logger.info("DISTURBANCE: %s generator tripped", d.target_id)

    def _maybe_inject_disturbance(self, pf) -> None:
        """Track grid health; inject a random disturbance once it stabilises."""
        max_loading = float(np.max(pf.line_loading_pct)) if len(pf.line_loading_pct) else 0
        has_v_viol = any(
            b.v_pu < 0.95 or b.v_pu > 1.05 for b in self._scenario.buses
        )
        is_healthy = max_loading < 80 and not has_v_viol

        if is_healthy:
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        if self._stable_steps >= self._stable_threshold:
            self._stable_steps = 0
            self._inject_random_disturbance()

    def _inject_random_disturbance(self) -> None:
        """Pick and apply a random grid disturbance to keep the agent busy."""
        options: list[str] = []

        closed_lines = [br for br in self._scenario.branches if br.is_closed]
        if len(closed_lines) > 2:          # keep at least 2 lines open-able
            options.append("line_trip")

        load_buses = [b for b in self._scenario.buses if b.load_mw > 20]
        if load_buses:
            options.append("load_spike")

        gen_buses = [
            b for b in self._scenario.buses
            if b.gen_mw > 30 and not b.is_slack
        ]
        if gen_buses:
            options.append("gen_reduction")

        if not options:
            return

        kind = _random.choice(options)
        event: dict = {"step": self._step_count, "hour": round(self._hour, 1)}

        if kind == "line_trip":
            target = _random.choice(closed_lines)
            target.is_closed = False
            self._solver.rebuild()
            event.update(kind="line_trip", target=target.branch_id)
            logger.info(
                "RANDOM DISTURBANCE step %d: %s tripped",
                self._step_count, target.branch_id,
            )
            # Auto-restore after ~2 hours (24 steps) so grid doesn't get stuck
            self._scenario.disturbances.append(
                Disturbance(
                    timestep=self._step_count + 24,
                    kind="line_restore",
                    target_id=target.branch_id,
                )
            )

        elif kind == "load_spike":
            target = _random.choice(load_buses)
            magnitude = _random.uniform(25, 55)
            target.load_mw += magnitude
            event.update(
                kind="load_spike", target=target.bus_id,
                magnitude=round(magnitude, 1),
            )
            logger.info(
                "RANDOM DISTURBANCE step %d: %s load +%.0f MW",
                self._step_count, target.bus_id, magnitude,
            )

        elif kind == "gen_reduction":
            target = _random.choice(gen_buses)
            reduction = target.gen_mw * _random.uniform(0.3, 0.6)
            target.gen_mw -= reduction
            event.update(
                kind="gen_reduction", target=target.bus_id,
                magnitude=round(reduction, 1),
            )
            logger.info(
                "RANDOM DISTURBANCE step %d: %s gen -%.0f MW",
                self._step_count, target.bus_id, reduction,
            )

        self.active_events.append(event)
        # Keep only the most recent 8 events
        if len(self.active_events) > 8:
            self.active_events = self.active_events[-8:]

    def _update_weather(self) -> None:
        """Update renewable output based on time-varying weather."""
        weather = self._scenario.weather.sample(self._hour % 24)
        self._curtailed_mw = 0.0

        for bus in self._scenario.buses:
            if not bus.is_renewable:
                continue
            if bus.gen_cost_per_mwh == 0 and "SOLAR" in bus.bus_id.upper():
                # Solar output proportional to irradiance
                potential = bus.gen_max_mw * weather["solar_w_m2"] / 1000.0
            elif bus.gen_cost_per_mwh == 0:
                # Wind output (cubic relationship, capped)
                ws = weather["wind_m_s"]
                potential = bus.gen_max_mw * min(1.0, (ws / 12.0) ** 3)
            else:
                potential = bus.gen_mw

            # If we must curtail
            actual = min(potential, bus.gen_max_mw)
            if actual < potential:
                self._curtailed_mw += potential - actual
            bus.gen_mw = max(bus.gen_min_mw, actual)

    def _apply_actions(self, action: dict[str, np.ndarray]) -> None:
        """Translate normalised [-1, 1] actions into physical MW changes."""
        # Redispatch generators
        redispatch = action.get("redispatch", np.zeros(len(self._gen_indices)))
        for i, bus_idx in enumerate(self._gen_indices):
            if i >= len(redispatch):
                break
            bus = self._scenario.buses[bus_idx]
            delta = float(redispatch[i]) * bus.gen_max_mw * 0.2  # max 20% swing per step
            new_gen = bus.gen_mw + delta
            bus.gen_mw = float(np.clip(new_gen, bus.gen_min_mw, bus.gen_max_mw))

        # Storage setpoints
        storage_act = action.get("storage", np.zeros(max(len(self._scenario.storage), 1)))
        dt_hours = 1.0 / self._scenario.steps_per_hour
        for i, s in enumerate(self._scenario.storage):
            if i >= len(storage_act):
                break
            setpoint_mw = float(storage_act[i]) * s.max_power_mw  # -1..1 → -max..+max
            # Update SoC (negative setpoint = charging)
            energy_mwh = -setpoint_mw * dt_hours * s.efficiency
            s.soc_mwh = float(np.clip(s.soc_mwh + energy_mwh, 0, s.capacity_mwh))
            # Remove previous injection before adding new one (prevent accumulation)
            prev_inj = self._prev_storage_inj.get(i, 0.0)
            self._scenario.buses[s.bus_idx].gen_mw -= prev_inj
            self._scenario.buses[s.bus_idx].gen_mw += setpoint_mw
            self._prev_storage_inj[i] = setpoint_mw

        # Load shaping
        load_shape = action.get("load_shape", np.zeros(max(len(self._flex_indices), 1)))
        for i, bus_idx in enumerate(self._flex_indices):
            if i >= len(load_shape):
                break
            bus = self._scenario.buses[bus_idx]
            delta = float(load_shape[i]) * bus.flexible_load_mw  # -1..1 of flexible range
            bus.load_mw = max(5.0, bus.load_mw + delta)  # floor: never shed below 5 MW

        # Topology switching
        topo = action.get("topology")
        if topo is not None:
            changed = False
            for i, br in enumerate(self._scenario.branches):
                if i >= len(topo):
                    break
                should_close = bool(topo[i])
                if br.is_closed != should_close:
                    br.is_closed = should_close
                    changed = True
            if changed:
                self._solver.rebuild()

    def _build_obs(self, pf) -> np.ndarray:
        """Flatten the grid state into a numeric observation vector."""
        obs_parts = []

        # Bus features
        for bus in self._scenario.buses:
            obs_parts.extend([bus.v_pu, bus.load_mw / 100.0, bus.gen_mw / 100.0])

        # Branch features
        for i, br in enumerate(self._scenario.branches):
            obs_parts.extend([
                pf.line_flows_mw[i] / 100.0,
                pf.line_loading_pct[i] / 100.0,
            ])

        # Storage features
        if self._scenario.storage:
            for s in self._scenario.storage:
                obs_parts.extend([
                    s.soc_mwh / s.capacity_mwh,
                    0.0,  # current setpoint placeholder
                ])
        else:
            obs_parts.extend([0.0, 0.0])

        # Weather
        weather = self._scenario.weather.sample(self._hour % 24)
        obs_parts.extend([
            weather["solar_w_m2"] / 1000.0,
            weather["wind_m_s"] / 20.0,
            weather["temperature_c"] / 50.0,
        ])

        # Time encoding
        hour_frac = (self._hour % 24) / 24.0
        obs_parts.extend([
            math.sin(2 * math.pi * hour_frac),
            math.cos(2 * math.pi * hour_frac),
        ])

        return np.array(obs_parts, dtype=np.float32)

    def _build_info(self, pf) -> dict[str, Any]:
        info = {
            "step": self._step_count,
            "hour": round(self._hour, 2),
            "converged": pf.converged,
            "curtailed_mw": self._curtailed_mw,
            "total_load_mw": sum(b.load_mw for b in self._scenario.buses),
            "total_gen_mw": sum(b.gen_mw for b in self._scenario.buses),
            "max_loading_pct": float(np.max(pf.line_loading_pct)) if len(pf.line_loading_pct) else 0,
            "alerts": self._collect_alerts(pf),
            "active_events": self.active_events[-5:],
        }
        return info

    def _collect_alerts(self, pf) -> list[str]:
        alerts = []
        for i, br in enumerate(self._scenario.branches):
            if pf.line_loading_pct[i] > 90:
                alerts.append(
                    f"{br.branch_id} loading {pf.line_loading_pct[i]:.0f}%"
                )
        for bus in self._scenario.buses:
            if bus.v_pu < 0.95:
                alerts.append(f"{bus.bus_id} voltage {bus.v_pu:.3f} pu (low)")
            elif bus.v_pu > 1.05:
                alerts.append(f"{bus.bus_id} voltage {bus.v_pu:.3f} pu (high)")
        return alerts

    # ── Growth mechanics ────────────────────────────────────

    def _maybe_growth_event(self) -> None:
        """Timer-based growth: every _growth_interval steps, pick a random event."""
        self._growth_timer += 1
        if self._growth_timer < self._growth_interval:
            return
        self._growth_timer = 0

        # Pick a random growth event type
        roll = _random.random()
        if roll < 0.30:
            self._growth_load_increase()
        elif roll < 0.55:
            self._growth_link_disconnect()
        elif roll < 0.80:
            self._growth_add_node()
        else:
            self._growth_add_link()

    def _growth_load_increase(self) -> None:
        """Randomly increase load at a bus."""
        load_buses = [b for b in self._scenario.buses if b.load_mw > 10]
        if not load_buses:
            return
        target = _random.choice(load_buses)
        increase = _random.uniform(15, 40)
        target.load_mw += increase
        msg = f"Load surge at {target.bus_id}: +{increase:.0f} MW (now {target.load_mw:.0f} MW)"
        self._add_chat_event("LOAD_INCREASE", target.bus_id, msg)
        self.active_events.append({
            "step": self._step_count, "hour": round(self._hour, 1),
            "kind": "load_spike", "target": target.bus_id,
            "magnitude": round(increase, 1),
        })

    def _growth_link_disconnect(self) -> None:
        """Trip a random closed line."""
        closed = [br for br in self._scenario.branches if br.is_closed]
        if len(closed) <= 3:  # keep minimum connectivity
            return
        target = _random.choice(closed)
        target.is_closed = False
        self._solver.rebuild()
        msg = f"Line {target.branch_id} tripped — link disconnected"
        self._add_chat_event("LINE_TRIP", target.branch_id, msg)
        self.active_events.append({
            "step": self._step_count, "hour": round(self._hour, 1),
            "kind": "line_trip", "target": target.branch_id,
        })
        # Auto-restore after 20 steps
        self._scenario.disturbances.append(
            Disturbance(timestep=self._step_count + 20, kind="line_restore",
                        target_id=target.branch_id)
        )

    def _growth_add_node(self) -> None:
        """Add a new bus + connecting branch to the grid."""
        if self._expansion_count >= 4:  # cap expansions per episode
            self._growth_load_increase()  # fallback
            return
        new_idx = self._next_bus_idx
        self._next_bus_idx += 1
        self._expansion_count += 1

        # Pick a random existing bus to connect to
        anchor = _random.choice(self._scenario.buses)
        # New bus is a load node
        names = ["INDUST", "CAMPUS", "HARBOR", "MARKET"]
        name = f"B{new_idx}-{names[self._expansion_count - 1]}"
        load = _random.uniform(30, 80)
        new_bus = Bus(
            bus_id=name, index=new_idx,
            load_mw=load, gen_mw=0, gen_max_mw=0,
            flexible_load_mw=load * 0.2,
        )
        self._scenario.buses.append(new_bus)
        # Update indices
        self._gen_indices = [b.index for b in self._scenario.buses if b.gen_max_mw > 0]
        self._flex_indices = [b.index for b in self._scenario.buses if b.flexible_load_mw > 0]

        branch_id = f"LN{anchor.index}{new_idx}"
        new_branch = Branch(
            branch_id=branch_id, from_idx=anchor.index, to_idx=new_idx,
            x_pu=_random.uniform(0.04, 0.08),
            capacity_mw=_random.uniform(60, 100),
        )
        self._scenario.branches.append(new_branch)

        # Rebuild solver with new topology
        self._solver = PowerFlowSolver(self._scenario.buses, self._scenario.branches)

        msg = f"New node {name} connected to {anchor.bus_id} via {branch_id} ({load:.0f} MW load)"
        self._add_chat_event("NODE_ADDED", name, msg)
        self.active_events.append({
            "step": self._step_count, "hour": round(self._hour, 1),
            "kind": "node_added", "target": name,
            "anchor": anchor.bus_id, "branch": branch_id,
        })

    def _growth_add_link(self) -> None:
        """Add a new branch between two existing buses that aren't directly connected."""
        buses = self._scenario.buses
        if len(buses) < 3:
            return
        # Find pairs not directly connected
        existing = {(br.from_idx, br.to_idx) for br in self._scenario.branches}
        existing |= {(br.to_idx, br.from_idx) for br in self._scenario.branches}
        candidates = []
        for i, a in enumerate(buses):
            for j, b in enumerate(buses):
                if i < j and (i, j) not in existing:
                    candidates.append((a, b))
        if not candidates:
            self._growth_load_increase()
            return
        a, b = _random.choice(candidates)
        branch_id = f"LN{a.index}{b.index}"
        new_branch = Branch(
            branch_id=branch_id, from_idx=a.index, to_idx=b.index,
            x_pu=_random.uniform(0.04, 0.08),
            capacity_mw=_random.uniform(60, 100),
        )
        self._scenario.branches.append(new_branch)
        self._solver = PowerFlowSolver(self._scenario.buses, self._scenario.branches)
        msg = f"New link {branch_id} connecting {a.bus_id} to {b.bus_id}"
        self._add_chat_event("LINK_ADDED", branch_id, msg)
        self.active_events.append({
            "step": self._step_count, "hour": round(self._hour, 1),
            "kind": "link_added", "target": branch_id,
            "from": a.bus_id, "to": b.bus_id,
        })

    def _add_chat_event(self, kind: str, target: str, message: str) -> None:
        """Add a system chat event for the frontend."""
        self.chat_events.append({
            "type": "system",
            "kind": kind,
            "target": target,
            "message": message,
            "step": self._step_count,
            "hour": round(self._hour, 1),
        })
        if len(self.chat_events) > 50:
            self.chat_events = self.chat_events[-50:]

    def render(self) -> str | None:
        if self.render_mode == "text":
            gs = self.get_grid_state()
            text = gs.to_context_block()
            if self._last_reward:
                text += f"\nREWARD: {self._last_reward.total:.2f}"
                text += f"  (cong={self._last_reward.congestion_penalty:.2f}"
                text += f"  volt={self._last_reward.voltage_penalty:.2f}"
                text += f"  curt={self._last_reward.curtailment_penalty:.2f}"
                text += f"  rel={self._last_reward.reliability_bonus:.2f})"
            print(text)
            return text
        return None
