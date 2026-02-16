#!/usr/bin/env python3
"""Conducto — Grid Pressure Autopilot

Continuous control loop that:
  1. Runs a physics-based grid simulation (DC power flow via GridEnv)
  2. Reads GridState from the simulator each timestep
  3. Asks Nemotron for recommended actions (redispatch, storage, topology)
  4. Translates LLM decisions into gym action arrays
  5. Steps the physics sim — gets real reward from power flow solver
  6. Records traces and outcomes for continuous learning
  7. Periodically fine-tunes the model on VESSL A100 when enough data accumulates
  8. Displays a live Rich terminal dashboard the whole time

Usage:
    python main.py                     # full loop with live UI
    python main.py --cycles 10         # run N cycles then stop
    python main.py --scenario 14-bus   # use IEEE 14-bus scenario
    python main.py --no-finetune       # skip VESSL fine-tuning triggers
    python main.py --fast              # 2-second intervals (demo mode)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ── Rich imports ──────────────────────────────────────────────
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.align import Align
from rich.rule import Rule

# ── Project imports ───────────────────────────────────────────
from nemotron_agent.config import Config
from nemotron_agent.agent import NemotronGridAgent, AgentDecision
from nemotron_agent.trace_buffer import Outcome
from nemotron_agent.grid_state import GridState
from nemotron_agent.gym import GridEnv, SCENARIOS

import numpy as np


# ── Constants ─────────────────────────────────────────────────
ONLINE_TRAIN_INTERVAL = 3       # do online weight update every N cycles
ONLINE_MIN_TRACES = 3           # min new traces needed for an update
ONLINE_LR = 1e-5                # learning rate for online LoRA updates
SAVE_CHECKPOINT_INTERVAL = 20   # save LoRA checkpoint every N online train calls
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
ADAPTERS_DIR = Path(__file__).parent / "adapters"
ADAPTERS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Session checkpoint — persist state across restarts
# ═══════════════════════════════════════════════════════════════
CHECKPOINT_FILE = CHECKPOINT_DIR / "session.json"


class SessionCheckpoint:
    """Saves and restores the agent's accumulated learning state.

    Persists:
      - Metrics history (rewards, latencies, reward windows)
      - Fine-tune run count and dataset sizes
      - Active model endpoint (base NIM vs self-hosted fine-tuned)
      - Adapter version info
    """

    @staticmethod
    def save(metrics: Metrics, active_endpoint: str, adapter_version: int) -> None:
        data = {
            "saved_at": time.time(),
            "cycle": metrics.cycle,
            "rewards": metrics.rewards,
            "latencies_ms": metrics.latencies_ms,
            "risk_levels": metrics.risk_levels,
            "action_counts": metrics.action_counts,
            "reward_windows": metrics.reward_windows,
            "finetune_runs": metrics.finetune_runs,
            "sft_dataset_size": metrics.sft_dataset_size,
            "rl_dataset_size": metrics.rl_dataset_size,
            "active_endpoint": active_endpoint,
            "adapter_version": adapter_version,
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load() -> dict | None:
        if not CHECKPOINT_FILE.exists():
            return None
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)

    @staticmethod
    def restore_metrics(metrics: Metrics, data: dict) -> None:
        """Restore metrics from a previous session."""
        metrics.rewards = data.get("rewards", [])
        metrics.latencies_ms = data.get("latencies_ms", [])
        metrics.risk_levels = data.get("risk_levels", [])
        metrics.action_counts = data.get("action_counts", [])
        metrics.reward_windows = data.get("reward_windows", [])
        metrics.finetune_runs = data.get("finetune_runs", 0)
        metrics.sft_dataset_size = data.get("sft_dataset_size", 0)
        metrics.rl_dataset_size = data.get("rl_dataset_size", 0)
        metrics.cycle = data.get("cycle", 0)


# ═══════════════════════════════════════════════════════════════
#  Metrics tracker — records everything for the dashboard
# ═══════════════════════════════════════════════════════════════
@dataclass
class Metrics:
    """Thread-safe metrics for the live dashboard."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Cycle stats
    cycle: int = 0
    start_time: float = field(default_factory=time.time)
    last_cycle_time: float = 0.0

    # Agent performance
    rewards: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    risk_levels: list[str] = field(default_factory=list)
    action_counts: list[int] = field(default_factory=list)
    assessments: list[str] = field(default_factory=list)

    # Grid health
    congestion_counts: list[int] = field(default_factory=list)
    voltage_violation_counts: list[int] = field(default_factory=list)
    total_load_mw: list[float] = field(default_factory=list)
    total_gen_mw: list[float] = field(default_factory=list)

    # Training
    traces_total: int = 0
    traces_with_outcome: int = 0
    finetune_runs: int = 0
    last_finetune_time: float = 0.0
    finetune_status: str = "idle"
    sft_dataset_size: int = 0
    rl_dataset_size: int = 0

    # Model improvement tracking (reward windowed averages)
    reward_windows: list[float] = field(default_factory=list)  # avg reward per 10-cycle window

    # Benchmark & rollback tracking
    baseline_avg_reward: float = -999.0   # best avg reward before swap
    post_swap_rewards: list[float] = field(default_factory=list)  # rewards since last endpoint swap
    swap_cycle: int = 0                    # cycle at which we last swapped
    rollback_count: int = 0               # how many times we rolled back
    benchmark_score: float = 0.0          # latest benchmark result
    using_finetuned: bool = False

    # Logs
    log_lines: list[str] = field(default_factory=list)

    def record_cycle(
        self,
        decision: AgentDecision,
        state: GridState,
        reward: float,
    ) -> None:
        with self._lock:
            self.cycle += 1
            self.last_cycle_time = time.time()
            self.rewards.append(reward)
            self.latencies_ms.append(decision.latency_ms)
            self.risk_levels.append(decision.risk_level)
            self.action_counts.append(len(decision.actions))
            self.assessments.append(decision.assessment[:120])
            self.congestion_counts.append(len(state.congested_lines))
            self.voltage_violation_counts.append(len(state.voltage_violations))
            self.total_load_mw.append(state.total_load_mw)
            self.total_gen_mw.append(state.total_generation_mw)

            # Compute windowed reward average every 10 cycles
            if self.cycle % 10 == 0 and len(self.rewards) >= 10:
                window_avg = statistics.mean(self.rewards[-10:])
                self.reward_windows.append(window_avg)

    def log(self, msg: str) -> None:
        with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_lines.append(f"[dim]{ts}[/dim]  {msg}")
            if len(self.log_lines) > 50:
                self.log_lines = self.log_lines[-50:]

    @property
    def avg_reward(self) -> float:
        return statistics.mean(self.rewards) if self.rewards else 0.0

    @property
    def recent_avg_reward(self) -> float:
        recent = self.rewards[-10:]
        return statistics.mean(recent) if recent else 0.0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def uptime_str(self) -> str:
        elapsed = int(time.time() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def reward_trend(self) -> str:
        """Arrow showing if model is improving."""
        if len(self.reward_windows) < 2:
            return "[dim]--[/dim]"
        recent = self.reward_windows[-1]
        prev = self.reward_windows[-2]
        diff = recent - prev
        if diff > 1.0:
            return "[bold green]++ improving[/bold green]"
        elif diff > 0.1:
            return "[green]+ improving[/green]"
        elif diff < -1.0:
            return "[bold red]-- declining[/bold red]"
        elif diff < -0.1:
            return "[red]- declining[/red]"
        return "[yellow]~ stable[/yellow]"


# ═══════════════════════════════════════════════════════════════
#  Rich Dashboard — builds the live terminal UI
# ═══════════════════════════════════════════════════════════════
class Dashboard:
    """Renders the Rich live dashboard."""

    RISK_COLORS = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold white on red",
        "unknown": "dim",
    }

    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics
        self.console = Console()

    def build(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=12),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="status", size=12),
            Layout(name="grid_health"),
        )
        layout["right"].split_column(
            Layout(name="agent"),
            Layout(name="model"),
        )

        layout["header"].update(self._header())
        layout["status"].update(self._status_panel())
        layout["grid_health"].update(self._grid_health_panel())
        layout["agent"].update(self._agent_panel())
        layout["model"].update(self._model_panel())
        layout["footer"].update(self._log_panel())

        return layout

    def _header(self) -> Panel:
        m = self.metrics
        title = Text()
        title.append("  CONDUCTO  ", style="bold white on blue")
        title.append("  Grid Pressure Autopilot  ", style="bold cyan")
        title.append(f"  cycle {m.cycle}  ", style="bold white on dark_green")
        title.append(f"  uptime {m.uptime_str}  ", style="dim")
        return Panel(Align.center(title), style="blue", box=box.DOUBLE)

    def _status_panel(self) -> Panel:
        m = self.metrics
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("key", style="bold cyan", width=18)
        table.add_column("value")

        table.add_row("Cycle", f"[bold]{m.cycle}[/bold]")
        table.add_row("Interval", f"{m.last_cycle_time - m.start_time:.0f}s elapsed" if m.cycle > 0 else "--")

        risk = m.risk_levels[-1] if m.risk_levels else "n/a"
        risk_style = self.RISK_COLORS.get(risk, "dim")
        table.add_row("Risk Level", f"[{risk_style}]{risk.upper()}[/{risk_style}]")

        table.add_row("Latency (avg)", f"{m.avg_latency:.0f} ms")
        table.add_row("Latency (p99)", f"{m.p99_latency:.0f} ms")
        table.add_row("Total Traces", f"{m.traces_total}")

        ft_style = "green" if "LIVE" in m.finetune_status else "bold yellow" if m.finetune_status != "idle" else "dim"
        table.add_row("Fine-tune", f"[{ft_style}]{m.finetune_status}[/{ft_style}]")

        return Panel(table, title="[bold]System Status[/bold]", border_style="cyan")

    def _grid_health_panel(self) -> Panel:
        m = self.metrics
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("key", style="bold green", width=18)
        table.add_column("value")

        load = m.total_load_mw[-1] if m.total_load_mw else 0
        gen = m.total_gen_mw[-1] if m.total_gen_mw else 0
        balance = gen - load
        bal_style = "green" if abs(balance) < 50 else "yellow" if abs(balance) < 100 else "red"

        table.add_row("Total Load", f"{load:.0f} MW")
        table.add_row("Total Gen", f"{gen:.0f} MW")
        table.add_row("Balance", f"[{bal_style}]{balance:+.0f} MW[/{bal_style}]")

        cong = m.congestion_counts[-1] if m.congestion_counts else 0
        cong_style = "green" if cong == 0 else "yellow" if cong < 3 else "red"
        table.add_row("Congested Lines", f"[{cong_style}]{cong}[/{cong_style}]")

        vv = m.voltage_violation_counts[-1] if m.voltage_violation_counts else 0
        vv_style = "green" if vv == 0 else "yellow" if vv < 2 else "red"
        table.add_row("Voltage Violations", f"[{vv_style}]{vv}[/{vv_style}]")

        # Sparkline for congestion over last 20 cycles
        spark = self._sparkline(m.congestion_counts[-20:], max_val=5)
        table.add_row("Congestion Trend", spark)

        return Panel(table, title="[bold]Grid Health[/bold]", border_style="green")

    def _agent_panel(self) -> Panel:
        m = self.metrics

        # Latest decision
        assessment = m.assessments[-1] if m.assessments else "Waiting for first cycle..."
        actions = m.action_counts[-1] if m.action_counts else 0
        risk = m.risk_levels[-1] if m.risk_levels else "n/a"
        risk_style = self.RISK_COLORS.get(risk, "dim")

        parts = []
        parts.append(f"[bold]Assessment:[/bold] {assessment}")
        parts.append(f"[bold]Risk:[/bold] [{risk_style}]{risk.upper()}[/{risk_style}]    [bold]Actions:[/bold] {actions}")
        parts.append("")

        # Recent decisions table
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("#", style="dim", width=5)
        table.add_column("Risk", width=10)
        table.add_column("Actions", width=8)
        table.add_column("Latency", width=10)
        table.add_column("Reward", width=10)
        table.add_column("Assessment", ratio=1)

        n = min(len(m.rewards), len(m.risk_levels), len(m.action_counts),
                len(m.latencies_ms), len(m.assessments))
        start = max(0, n - 8)
        for i in range(start, n):
            r = m.risk_levels[i]
            rs = self.RISK_COLORS.get(r, "dim")
            rew = m.rewards[i]
            rew_style = "green" if rew > 0 else "red" if rew < -5 else "yellow"
            table.add_row(
                str(i + 1),
                f"[{rs}]{r}[/{rs}]",
                str(m.action_counts[i]),
                f"{m.latencies_ms[i]:.0f}ms",
                f"[{rew_style}]{rew:+.1f}[/{rew_style}]",
                m.assessments[i][:60],
            )

        content = Text.from_markup("\n".join(parts))

        inner = Table.grid()
        inner.add_row(content)
        inner.add_row(table)

        return Panel(inner, title="[bold]Agent Decisions[/bold]", border_style="magenta")

    def _model_panel(self) -> Panel:
        m = self.metrics
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("key", style="bold yellow", width=22)
        table.add_column("value")

        table.add_row("Avg Reward (all)", f"{m.avg_reward:+.2f}")
        table.add_row("Avg Reward (last 10)", f"[bold]{m.recent_avg_reward:+.2f}[/bold]")
        table.add_row("Trend", m.reward_trend)
        table.add_row("Fine-tune Runs", f"{m.finetune_runs}")
        table.add_row("Benchmark Score", f"[bold]{m.benchmark_score:.0f}/100[/bold]" if m.benchmark_score > 0 else "[dim]--[/dim]")
        table.add_row("Rollbacks", f"[red]{m.rollback_count}[/red]" if m.rollback_count > 0 else "[green]0[/green]")
        model_label = "[bold green]fine-tuned[/bold green]" if m.using_finetuned else "[cyan]cloud NIM[/cyan]"
        table.add_row("Active Model", model_label)
        table.add_row("SFT Dataset Size", f"{m.sft_dataset_size}")
        table.add_row("", "")

        # Reward sparkline over 10-cycle windows
        if m.reward_windows:
            spark = self._reward_sparkline(m.reward_windows[-20:])
            table.add_row("Reward History", spark)
        else:
            table.add_row("Reward History", "[dim]collecting data...[/dim]")

        # Reward bar chart for last 20 cycles
        table.add_row("", "")
        table.add_row("[bold]Per-cycle Rewards[/bold]", "")
        recent = m.rewards[-20:]
        if recent:
            bar = self._reward_bar(recent)
            table.add_row("", bar)

        return Panel(table, title="[bold]Model Improvement[/bold]", border_style="yellow")

    def _log_panel(self) -> Panel:
        m = self.metrics
        lines = m.log_lines[-10:]
        content = "\n".join(lines) if lines else "[dim]Starting up...[/dim]"
        return Panel(
            Text.from_markup(content),
            title="[bold]Live Log[/bold]",
            border_style="dim",
        )

    # ── Visualisation helpers ─────────────────────────────
    @staticmethod
    def _sparkline(values: list[int | float], max_val: float | None = None) -> str:
        if not values:
            return "[dim]...[/dim]"
        blocks = " ▁▂▃▄▅▆▇█"
        mx = max_val or max(values) or 1
        chars = []
        for v in values:
            idx = int(min(v / mx, 1.0) * (len(blocks) - 1))
            chars.append(blocks[idx])
        return "".join(chars)

    @staticmethod
    def _reward_sparkline(windows: list[float]) -> Text:
        if not windows:
            return Text("...")
        # Normalize to -10..+10 range
        result = Text()
        blocks = "▁▂▃▄▅▆▇█"
        for val in windows:
            norm = max(0, min((val + 10) / 20, 1.0))
            idx = int(norm * (len(blocks) - 1))
            color = "green" if val > 0 else "red" if val < -2 else "yellow"
            result.append(blocks[idx], style=color)
        return result

    @staticmethod
    def _reward_bar(rewards: list[float]) -> Text:
        result = Text()
        for r in rewards:
            if r > 5:
                result.append("█", style="bold green")
            elif r > 0:
                result.append("▆", style="green")
            elif r > -2:
                result.append("▃", style="yellow")
            elif r > -5:
                result.append("▂", style="red")
            else:
                result.append("▁", style="bold red")
        return result


# ═══════════════════════════════════════════════════════════════
#  Action translator — LLM JSON → gym action dict
# ═══════════════════════════════════════════════════════════════
class ActionTranslator:
    """Translates the agent's JSON actions into the gym's action dict.

    The LLM outputs actions like:
        {"type": "REDISPATCH", "target": "B2-GAS", "params": {"target_mw": 150}}
        {"type": "STORAGE",   "target": "BESS-SUBURB", "params": {"setpoint_mw": 30}}
        {"type": "TOPOLOGY",  "target": "LN01", "params": {"action": "open"}}

    The gym expects normalised arrays:
        {"redispatch": [-1..1], "storage": [-1..1], "load_shape": [-1..1], "topology": [0/1]}
    """

    def __init__(self, env: GridEnv) -> None:
        self.env = env
        scenario = env._base_scenario
        # Build lookup maps from bus/branch IDs → indices
        self._gen_buses = [b for b in scenario.buses if b.gen_max_mw > 0]
        self._gen_id_to_idx = {b.bus_id: i for i, b in enumerate(self._gen_buses)}
        self._storage_units = scenario.storage
        self._storage_id_to_idx = {s.unit_id: i for i, s in enumerate(scenario.storage)}
        self._flex_buses = [b for b in scenario.buses if b.flexible_load_mw > 0]
        self._flex_id_to_idx = {b.bus_id: i for i, b in enumerate(self._flex_buses)}
        self._branches = scenario.branches
        self._branch_id_to_idx = {br.branch_id: i for i, br in enumerate(scenario.branches)}

    def translate(self, decision: AgentDecision) -> dict[str, np.ndarray]:
        """Convert LLM decision actions → gym action dict."""
        n_gen = len(self._gen_buses)
        n_storage = max(len(self._storage_units), 1)
        n_flex = max(len(self._flex_buses), 1)
        n_branch = len(self._branches)

        redispatch = np.zeros(n_gen, dtype=np.float32)
        storage = np.zeros(n_storage, dtype=np.float32)
        load_shape = np.zeros(n_flex, dtype=np.float32)
        topology = np.ones(n_branch, dtype=np.int8)  # default: all closed

        for action in decision.actions:
            a_type = action.get("type", "").upper()
            target = action.get("target", "")
            params = action.get("params", {})

            if a_type == "REDISPATCH":
                idx = self._gen_id_to_idx.get(target)
                if idx is not None:
                    bus = self._gen_buses[idx]
                    target_mw = params.get("target_mw", bus.gen_mw)
                    delta = target_mw - bus.gen_mw
                    # Normalise to [-1, 1] where 1 = 20% of max capacity
                    redispatch[idx] = np.clip(delta / (bus.gen_max_mw * 0.2), -1, 1)

            elif a_type == "STORAGE":
                idx = self._storage_id_to_idx.get(target)
                if idx is not None:
                    unit = self._storage_units[idx]
                    setpoint = params.get("setpoint_mw", 0)
                    # Normalise to [-1, 1] where 1 = max discharge
                    storage[idx] = np.clip(setpoint / unit.max_power_mw, -1, 1)

            elif a_type == "LOAD_SHAPE":
                idx = self._flex_id_to_idx.get(target)
                if idx is not None:
                    bus = self._flex_buses[idx]
                    delta_mw = params.get("delta_mw", 0)
                    load_shape[idx] = np.clip(delta_mw / bus.flexible_load_mw, -1, 1) if bus.flexible_load_mw > 0 else 0

            elif a_type == "TOPOLOGY":
                idx = self._branch_id_to_idx.get(target)
                if idx is not None:
                    action_str = params.get("action", "close")
                    topology[idx] = 0 if action_str == "open" else 1

        return {
            "redispatch": redispatch,
            "storage": storage,
            "load_shape": load_shape,
            "topology": topology,
        }


# ═══════════════════════════════════════════════════════════════
#  Outcome builder — gym step info → trace Outcome
# ═══════════════════════════════════════════════════════════════
def build_outcome_from_gym(
    reward: float,
    info: dict,
    prev_info: dict | None,
) -> Outcome:
    """Convert gym step results into an Outcome for the trace buffer."""
    prev_congested = len(prev_info.get("alerts", [])) if prev_info else 0
    curr_congested = len(info.get("alerts", []))

    return Outcome(
        congestion_delta=curr_congested - prev_congested,
        voltage_violations_delta=0,  # embedded in reward signal
        curtailment_mw_delta=info.get("curtailed_mw", 0),
        cost_delta=0.0,
        reward=reward,
    )


# ═══════════════════════════════════════════════════════════════
#  A100 Server + Online Training
# ═══════════════════════════════════════════════════════════════
_ssh_tunnel_proc = None       # SSH tunnel to VESSL inference server
_vessl_runner = None          # VESSLRunner singleton for the session
_cloud_endpoint: str = ""     # saved cloud NIM endpoint for fallback
_online_train_count: int = 0  # number of online train calls made
_last_train_cycle: int = 0    # cycle at which we last trained

# Step status markers for the init UI
_STEP_PENDING = "[dim]     [/dim]"
_STEP_ACTIVE  = "[bold yellow] >>> [/bold yellow]"
_STEP_DONE    = "[bold green] [OK] [/bold green]"
_STEP_FAIL    = "[bold red] [!!] [/bold red]"


def _build_init_panel(steps: list[tuple[str, str]], status_msg: str, log_lines: list[str] | None = None) -> Panel:
    """Build the Rich panel for A100 initialization progress."""
    rows = []
    for marker, label in steps:
        rows.append(f"  {marker}  {label}")
    rows.append("")
    rows.append(f"  [dim]{status_msg}[/dim]")

    # Show streaming log lines from the remote server
    if log_lines:
        rows.append("")
        rows.append("  [dim]─── remote log ───[/dim]")
        for line in log_lines[-4:]:
            # Truncate long lines and dim them
            clean = line.strip()[:100]
            if clean:
                rows.append(f"  [dim]{clean}[/dim]")

    content = "\n".join(rows)
    return Panel(
        Text.from_markup(content),
        title="[bold white on blue]  CONDUCTO  [/bold white on blue]  [bold cyan]Initialising A100 GPU Server[/bold cyan]",
        border_style="blue",
        box=box.DOUBLE,
        padding=(1, 2),
    )


def init_a100_server(metrics: Metrics, console: Console) -> bool:
    """Deploy inference+training server on A100 and open SSH tunnel.

    Shows a Rich progress panel during initialization.
    Called once at startup before the main loop begins.
    Returns True if the A100 is ready for inference and online training.
    """
    global _ssh_tunnel_proc, _vessl_runner

    metrics.finetune_status = "deploying to A100"

    step_labels = [
        "Connect to A100 via SSH",
        "Setup workspace & dependencies",
        "Upload inference + training server",
        "Start model server on GPU",
        "Open SSH tunnel (localhost:9090)",
        "Load model into GPU memory",
    ]
    step_status = [_STEP_PENDING] * len(step_labels)

    remote_log_lines: list[str] = []

    def _render(active_idx: int = -1, msg: str = ""):
        steps = []
        for i, label in enumerate(step_labels):
            marker = step_status[i]
            if i == active_idx and marker == _STEP_PENDING:
                marker = _STEP_ACTIVE
            style = "bold" if i == active_idx else ""
            steps.append((marker, f"[{style}]{label}[/{style}]" if style else label))
        console.clear()
        console.print()
        console.print(_build_init_panel(steps, msg, remote_log_lines if remote_log_lines else None))

    try:
        # Step 0: Connect
        _render(0, "Connecting to VESSL A100...")
        from nemotron_agent.training.vessl_runner import VESSLRunner
        _vessl_runner = VESSLRunner()
        step_status[0] = _STEP_DONE

        # Check if server already running (from previous session)
        if _ssh_tunnel_proc is not None and _ssh_tunnel_proc.poll() is None:
            _render(0, "Checking existing server...")
            if _vessl_runner.wait_for_server(local_port=9090, timeout_sec=10):
                for i in range(len(step_status)):
                    step_status[i] = _STEP_DONE
                _render(-1, "A100 server already running from previous session!")
                time.sleep(1)
                metrics.finetune_status = "A100 LIVE (online)"
                metrics.using_finetuned = True
                metrics.log("[bold green]A100 server already running — reusing[/bold green]")
                return True

        # Step 1: Setup workspace
        _render(1, "Creating workspace dirs & installing deps (transformers, peft, trl, bitsandbytes)...")
        _vessl_runner.setup_workspace()
        step_status[1] = _STEP_DONE

        # Step 2: Upload serve script
        _render(2, "Uploading peft_serve.py to A100...")
        adapter_path = f"{_vessl_runner._remote_dir}/output/sft-adapters"
        step_status[2] = _STEP_DONE

        # Step 3: Start server (non-blocking — fires and moves on)
        _render(3, "Launching peft_serve.py on A100 (nohup background)...")
        _vessl_runner.deploy_inference_server(
            adapter_path=adapter_path,
            port=8000,
        )
        step_status[3] = _STEP_DONE
        _render(3, "Server process launched on A100")

        # Step 4: SSH tunnel
        _render(4, "Opening SSH tunnel localhost:9090 → A100:8000...")
        if _ssh_tunnel_proc is not None:
            try:
                _ssh_tunnel_proc.terminate()
                _ssh_tunnel_proc.wait(timeout=5)
            except Exception:
                pass
        _ssh_tunnel_proc = _vessl_runner.start_ssh_tunnel(
            remote_port=8000, local_port=9090,
        )
        step_status[4] = _STEP_DONE

        # Step 5: Wait for model to load — stream remote log for visibility
        import urllib.request
        url = "http://localhost:9090/health"
        start = time.time()
        timeout_sec = 180
        while time.time() - start < timeout_sec:
            elapsed = int(time.time() - start)

            # Tail remote serve.log for streaming progress
            try:
                log_tail = _vessl_runner.tail_server_log(lines=4)
                if log_tail:
                    remote_log_lines.clear()
                    remote_log_lines.extend(log_tail.split("\n"))
            except Exception:
                pass

            # Build status message from latest log line
            last_line = remote_log_lines[-1].strip() if remote_log_lines else ""
            if "Loading base model" in last_line:
                status = f"Downloading & loading base model... ({elapsed}s)"
            elif "Loading LoRA" in last_line:
                status = f"Loading LoRA adapters... ({elapsed}s)"
            elif "Model ready" in last_line:
                status = f"Model loaded! Starting server... ({elapsed}s)"
            elif "creating fresh LoRA" in last_line.lower():
                status = f"Creating fresh LoRA adapters... ({elapsed}s)"
            else:
                status = f"Loading Nemotron-Mini-4B into GPU memory... ({elapsed}s)"

            _render(5, status)

            try:
                req = urllib.request.Request(url, method="GET")
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    step_status[5] = _STEP_DONE
                    remote_log_lines.clear()
                    _render(-1, "A100 GPU server is LIVE and ready!")
                    time.sleep(1)
                    metrics.finetune_status = "A100 LIVE (online)"
                    metrics.using_finetuned = True
                    metrics.log("[bold green]A100 inference+training server is LIVE[/bold green]")
                    return True
            except Exception:
                pass
            time.sleep(3)

        # Timeout — show last log lines for debugging
        step_status[5] = _STEP_FAIL
        _render(-1, f"Model failed to load within {timeout_sec}s — check serve.log on A100")
        time.sleep(2)
        metrics.finetune_status = "A100 deploy failed"
        metrics.log("[red]A100 server failed to start — model load timeout[/red]")
        return False

    except Exception as e:
        # Mark current step as failed
        for i, s in enumerate(step_status):
            if s == _STEP_PENDING or s == _STEP_ACTIVE:
                step_status[i] = _STEP_FAIL
                break
        _render(-1, f"Error: {e}")
        time.sleep(2)
        metrics.finetune_status = "A100 init error"
        metrics.log(f"[red]A100 init failed:[/red] {e}")
        return False


def do_online_train(
    metrics: Metrics,
    trace_buffer,
    cycle: int,
) -> None:
    """Send recent good traces to A100 for rapid LoRA weight updates.

    Called every ONLINE_TRAIN_INTERVAL cycles from the main loop.
    Runs synchronously but fast (~2-5s for a handful of traces).
    Never raises — training errors are logged and swallowed so the
    main control loop keeps running.
    """
    global _online_train_count, _last_train_cycle

    if _vessl_runner is None:
        return
    if cycle - _last_train_cycle < ONLINE_TRAIN_INTERVAL:
        return

    try:
        # Collect recent traces with outcomes (good ones only for quality)
        recent_traces = trace_buffer.sample_traces(
            ONLINE_TRAIN_INTERVAL * 2,
            only_with_outcome=True,
            min_reward=-10.0,  # skip catastrophic failures
        )
        if len(recent_traces) < ONLINE_MIN_TRACES:
            metrics.log("[dim]  train: not enough good traces yet, skipping[/dim]")
            _last_train_cycle = cycle  # avoid retrying every cycle
            return

        # Sort by reward, keep the better half for training
        recent_traces = sorted(recent_traces, key=lambda t: t.get("reward", -999), reverse=True)
        good_traces = recent_traces[:max(ONLINE_MIN_TRACES, len(recent_traces) // 2)]

        # Format for the /train endpoint
        train_data = []
        for t in good_traces:
            if "messages" in t and "raw_response" in t:
                train_data.append({
                    "messages": t["messages"],
                    "raw_response": t["raw_response"],
                })

        if not train_data:
            _last_train_cycle = cycle
            return

        metrics.log(
            f"[bold yellow]Online train[/bold yellow] — "
            f"sending {len(train_data)} traces to A100..."
        )
        metrics.finetune_status = "online training..."

        result = _vessl_runner.online_train(
            traces=train_data,
            local_port=9090,
            learning_rate=ONLINE_LR,
        )

        _last_train_cycle = cycle
        _online_train_count += 1

        if result.get("status") == "ok":
            steps = result.get("steps", 0)
            loss = result.get("loss", 0)
            total = result.get("total_steps", 0)
            train_ms = result.get("train_ms", 0)
            metrics.finetune_runs = total  # total gradient steps
            metrics.finetune_status = f"A100 LIVE — {total} steps, loss={loss:.3f}"
            metrics.log(
                f"  [green]Done:[/green] {steps} steps, loss={loss:.4f}, "
                f"total={total} steps, {train_ms:.0f}ms"
            )

            # Save checkpoint periodically
            if _online_train_count % SAVE_CHECKPOINT_INTERVAL == 0:
                metrics.log("  [cyan]Saving LoRA checkpoint on A100...[/cyan]")
                _vessl_runner.save_checkpoint(local_port=9090)
        else:
            error = result.get("error", "unknown")
            metrics.log(f"  [red]Online train failed:[/red] {error}")
            metrics.finetune_status = f"A100 LIVE (train error: {error})"

    except Exception as e:
        metrics.log(f"  [red]Online train error:[/red] {e}")
        metrics.finetune_status = "A100 LIVE (train error)"
        _last_train_cycle = cycle  # don't retry immediately


# ═══════════════════════════════════════════════════════════════
#  File logger — writes structured JSON logs
# ═══════════════════════════════════════════════════════════════
class FileLogger:
    """Writes structured JSON log entries to disk."""

    def __init__(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"conducto_run_{ts}.jsonl"

    def log_cycle(
        self,
        cycle: int,
        state: GridState,
        decision: AgentDecision,
        outcome: Outcome,
    ) -> None:
        entry = {
            "cycle": cycle,
            "timestamp": time.time(),
            "grid": {
                "load_mw": state.total_load_mw,
                "gen_mw": state.total_generation_mw,
                "congested_lines": len(state.congested_lines),
                "voltage_violations": len(state.voltage_violations),
            },
            "decision": {
                "risk_level": decision.risk_level,
                "n_actions": len(decision.actions),
                "latency_ms": decision.latency_ms,
                "assessment": decision.assessment,
                "actions": decision.actions,
            },
            "outcome": {
                "congestion_delta": outcome.congestion_delta,
                "voltage_delta": outcome.voltage_violations_delta,
                "curtailment_delta": outcome.curtailment_mw_delta,
                "cost_delta": outcome.cost_delta,
                "reward": outcome.reward,
            },
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════
def run(args: argparse.Namespace) -> None:
    console = Console()
    metrics = Metrics()
    dashboard = Dashboard(metrics)
    file_logger = FileLogger()

    metrics.log("[bold blue]Initialising Conducto...[/bold blue]")

    # ── Init components ───────────────────────────────────
    cfg = Config()
    agent = NemotronGridAgent(cfg)

    # The gym IS the grid — DC power flow, disturbances, physics-based rewards
    scenario_fn = SCENARIOS.get(args.scenario, SCENARIOS["6-bus"])
    env = GridEnv(scenario=scenario_fn())
    translator = ActionTranslator(env)
    obs, env_info = env.reset()

    interval = 2.0 if args.fast else cfg.agent.loop_interval_sec
    max_cycles = args.cycles or float("inf")
    episode_steps = int(env._base_scenario.episode_hours * env._base_scenario.steps_per_hour)

    # Track adapter version (mutable list so finetune fn can update it)
    adapter_version = [0]
    active_endpoint = cfg.nvidia.base_url
    global _cloud_endpoint
    _cloud_endpoint = cfg.nvidia.base_url  # save for fallback

    # ── Restore from previous session if available ────────
    checkpoint = SessionCheckpoint.load()
    if checkpoint and not args.fresh:
        SessionCheckpoint.restore_metrics(metrics, checkpoint)
        adapter_version[0] = checkpoint.get("adapter_version", 0)
        metrics.log(
            f"[green]Restored session:[/green] {metrics.cycle} prior cycles"
        )
    else:
        metrics.log("[dim]No previous session found — starting fresh[/dim]")

    # ── Initialise A100 inference+training server ────────
    # This is the key change: the A100 is ready BEFORE the main loop
    if not args.no_finetune:
        a100_ready = init_a100_server(metrics, console)
        if a100_ready:
            # Swap agent to use A100 from cycle 1
            a100_url = "http://localhost:9090/v1"
            from nemotron_agent.nim_client import NIMClient
            from nemotron_agent.config import NVIDIAConfig
            agent.nim = NIMClient(NVIDIAConfig(
                api_key=cfg.nvidia.api_key,
                base_url=a100_url,
                model_id="nemotron-finetuned",
            ))
            active_endpoint = a100_url
            metrics.log(
                f"[bold green]Agent using A100 model[/bold green] at [cyan]{a100_url}[/cyan]"
            )
        else:
            metrics.log(
                "[yellow]A100 not available — using cloud NIM endpoint[/yellow]"
            )

    metrics.log(f"Agent ready — model: [cyan]{cfg.nvidia.model_id}[/cyan]")
    metrics.log(f"Endpoint: [cyan]{active_endpoint}[/cyan]")
    metrics.log(f"Loop interval: [cyan]{interval}s[/cyan]   Online training: [cyan]{'enabled' if not args.no_finetune else 'disabled'}[/cyan]")
    metrics.log(f"Logging to: [dim]{file_logger.path}[/dim]")
    metrics.log("")

    prev_outcome: Outcome | None = None
    prev_info: dict | None = env_info if isinstance(env_info, dict) else {}
    episode_num = 1
    step_in_episode = 0

    # ── Web dashboard (optional) ─────────────────────────
    web_state = None
    if args.web:
        from frontend.server import dashboard_state, run_server, start_vessl_monitor
        web_state = dashboard_state
        # Start VESSL GPU monitor
        if cfg.vessl.ssh_host:
            start_vessl_monitor(cfg.vessl.ssh_host, cfg.vessl.ssh_key_path, cfg.vessl.ssh_port)
        # Start web server in background thread
        web_thread = threading.Thread(
            target=run_server, kwargs={"port": args.web_port}, daemon=True, name="web-dashboard",
        )
        web_thread.start()
        metrics.log(f"[bold green]Web dashboard:[/bold green] [link=http://localhost:{args.web_port}]http://localhost:{args.web_port}[/link]")
        # Push initial topology info
        web_state.update({"topology": {"name": args.scenario}})

    metrics.log(f"Gym environment ready — scenario [cyan]{args.scenario}[/cyan], "
                f"episode length {episode_steps} steps")

    with Live(dashboard.build(), console=console, refresh_per_second=2, screen=True) as live:
        cycle = 0
        while cycle < max_cycles:
            try:
                # ── 1. Read grid state ────────────────────────
                metrics.log(
                    f"[bold]--- Cycle {cycle + 1} ---[/bold]  "
                    f"ep={episode_num}  step={step_in_episode}"
                )
                state = env.get_grid_state()
                cong = len(state.congested_lines)
                vv = len(state.voltage_violations)
                cong_s = "green" if cong == 0 else "yellow" if cong < 3 else "red"
                metrics.log(
                    f"  [dim]grid:[/dim] load={state.total_load_mw:.0f}MW  "
                    f"gen={state.total_generation_mw:.0f}MW  "
                    f"[{cong_s}]cong={cong}[/{cong_s}]  viol={vv}"
                )

                # ── 2. Query agent ────────────────────────────
                metrics.log("  [dim]agent:[/dim] querying Nemotron...")
                decision = agent.decide(state, outcome=prev_outcome)
                metrics.log(
                    f"  [dim]agent:[/dim] responded in {decision.latency_ms:.0f}ms  "
                    f"risk=[bold]{decision.risk_level}[/bold]  "
                    f"actions={len(decision.actions)}"
                )

                # ── 3. Translate actions ──────────────────────
                gym_action = translator.translate(decision)
                action_types = [a.get("type", "?") for a in decision.actions]
                metrics.log(
                    f"  [dim]translate:[/dim] {', '.join(action_types) or 'no-op'}"
                )

                # ── 4. Step physics sim ───────────────────────
                obs, reward, terminated, truncated, info = env.step(gym_action)
                step_in_episode += 1
                alerts = info.get("alerts", [])
                reward_style = "green" if reward > 0 else "red" if reward < -5 else "yellow"
                metrics.log(
                    f"  [dim]physics:[/dim] [{reward_style}]reward={reward:+.1f}[/{reward_style}]  "
                    f"alerts={len(alerts)}  "
                    f"{'[red]TERMINATED[/red]' if terminated else '[green]ok[/green]'}"
                )

                # ── 5. Build outcome ──────────────────────────
                outcome = build_outcome_from_gym(reward, info, prev_info)

                # ── 6. Record metrics ─────────────────────────
                metrics.record_cycle(decision, state, outcome.reward)
                metrics.traces_total = agent.traces.size

                # ── 7. Log to file ────────────────────────────
                file_logger.log_cycle(cycle + 1, state, decision, outcome)

                # ── 8. Assessment summary ─────────────────────
                metrics.log(
                    f"  [dim]assessment:[/dim] {decision.assessment[:100]}"
                )

                # ── 9. Episode boundary ───────────────────────
                if terminated or truncated:
                    ep_rewards = metrics.rewards[-step_in_episode:] if step_in_episode > 0 else [0]
                    metrics.log(
                        f"  [bold cyan]Episode {episode_num} done[/bold cyan] "
                        f"({step_in_episode} steps, avg reward "
                        f"{statistics.mean(ep_rewards):+.2f})"
                    )
                    obs, env_info = env.reset()
                    prev_info = env_info if isinstance(env_info, dict) else {}
                    episode_num += 1
                    step_in_episode = 0
                else:
                    prev_info = info

                # ── 10. Online training ───────────────────────
                if not args.no_finetune and metrics.using_finetuned:
                    do_online_train(metrics, agent.traces, cycle)

                # 11. Update dashboard
                live.update(dashboard.build())

                # 12. Push to web dashboard (if running)
                if web_state is not None:
                    # Build recent decisions list safely
                    n_recent = min(8, len(metrics.rewards), len(metrics.risk_levels),
                                   len(metrics.action_counts), len(metrics.latencies_ms),
                                   len(metrics.assessments))
                    recent_decisions = []
                    for i in range(n_recent):
                        idx = len(metrics.rewards) - n_recent + i
                        recent_decisions.append({
                            "cycle": idx + 1,
                            "risk_level": metrics.risk_levels[idx],
                            "assessment": metrics.assessments[idx][:120],
                            "n_actions": metrics.action_counts[idx],
                            "reward": metrics.rewards[idx],
                            "latency_ms": metrics.latencies_ms[idx],
                        })

                    # MERGE vessl data — don't overwrite GPU monitor readings
                    current_vessl = web_state.get().get("vessl", {})
                    current_vessl["finetune_status"] = metrics.finetune_status
                    current_vessl["finetune_runs"] = metrics.finetune_runs

                    # Include active grid events (disturbances) for frontend
                    grid_data = state.to_dict()
                    grid_data["active_events"] = getattr(env, "active_events", [])[-5:]

                    web_state.update({
                        "cycle": cycle + 1,
                        "episode": episode_num,
                        "step_in_episode": step_in_episode,
                        "grid": grid_data,
                        "agent": {
                            "recent_decisions": recent_decisions,
                            "latest_actions": decision.actions,
                            "latest_assessment": decision.assessment[:200],
                            "latest_risk": decision.risk_level,
                        },
                        "metrics": {
                            "rewards": metrics.rewards[-30:],
                            "avg_reward": metrics.avg_reward,
                            "recent_avg_reward": metrics.recent_avg_reward,
                            "avg_latency": metrics.avg_latency,
                            "reward_trend": "improving" if len(metrics.reward_windows) >= 2 and metrics.reward_windows[-1] > metrics.reward_windows[-2] else "stable" if len(metrics.reward_windows) < 2 else "declining",
                            "traces_total": metrics.traces_total,
                            "online_train_steps": metrics.finetune_runs,
                            "using_finetuned": metrics.using_finetuned,
                            "finetune_status": metrics.finetune_status,
                        },
                        "vessl": current_vessl,
                    })

                # Save for next cycle
                prev_outcome = outcome
                cycle += 1

                # Wait for next cycle
                time.sleep(interval)

            except KeyboardInterrupt:
                metrics.log("[bold red]Interrupted by user[/bold red]")
                break
            except Exception as e:
                metrics.log(f"[bold red]Error:[/bold red] {e}")
                time.sleep(interval)
                cycle += 1

    # ── Clean up SSH tunnel ─────────────────────────────────
    if _ssh_tunnel_proc is not None:
        _ssh_tunnel_proc.terminate()

    # ── Save checkpoint ────────────────────────────────────
    SessionCheckpoint.save(metrics, active_endpoint, adapter_version[0])

    # ── Shutdown summary ──────────────────────────────────
    console.print()
    console.print(Rule("[bold blue]Conducto Session Summary[/bold blue]"))
    console.print()

    summary_table = Table(box=box.ROUNDED, title="Run Statistics", title_style="bold cyan")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Cycles", str(metrics.cycle))
    summary_table.add_row("Uptime", metrics.uptime_str)
    summary_table.add_row("Total Traces", str(metrics.traces_total))
    summary_table.add_row("Avg Reward (all)", f"{metrics.avg_reward:+.2f}")
    summary_table.add_row("Avg Reward (last 10)", f"{metrics.recent_avg_reward:+.2f}")
    summary_table.add_row("Avg Latency", f"{metrics.avg_latency:.0f} ms")
    summary_table.add_row("P99 Latency", f"{metrics.p99_latency:.0f} ms")
    summary_table.add_row("Fine-tune Runs", str(metrics.finetune_runs))
    summary_table.add_row("Adapter Version", f"v{adapter_version[0]}" if adapter_version[0] > 0 else "base model")
    summary_table.add_row("Active Endpoint", active_endpoint)
    summary_table.add_row("SFT Dataset Size", str(metrics.sft_dataset_size))
    summary_table.add_row("Log File", str(file_logger.path))
    summary_table.add_row("Checkpoint", str(CHECKPOINT_FILE))
    console.print(summary_table)

    # Model improvement summary
    if len(metrics.reward_windows) >= 2:
        console.print()
        improvement_table = Table(box=box.ROUNDED, title="Model Improvement Tracking", title_style="bold yellow")
        improvement_table.add_column("Window", style="dim")
        improvement_table.add_column("Avg Reward", justify="right")
        improvement_table.add_column("Trend", justify="center")

        for i, w in enumerate(metrics.reward_windows):
            trend = ""
            if i > 0:
                diff = w - metrics.reward_windows[i - 1]
                if diff > 0.5:
                    trend = "[green]^[/green]"
                elif diff < -0.5:
                    trend = "[red]v[/red]"
                else:
                    trend = "[yellow]=[/yellow]"
            rew_style = "green" if w > 0 else "red" if w < -2 else "yellow"
            improvement_table.add_row(
                f"Cycles {i * 10 + 1}-{(i + 1) * 10}",
                f"[{rew_style}]{w:+.2f}[/{rew_style}]",
                trend,
            )
        console.print(improvement_table)

    console.print()
    console.print(f"[dim]Full logs saved to:[/dim] [bold]{file_logger.path}[/bold]")
    console.print()


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conducto — Grid Pressure Autopilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Number of cycles to run (0 = infinite, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--no-finetune", action="store_true",
        help="Disable automatic VESSL fine-tuning triggers",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: 2-second intervals for quick demos",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore previous session checkpoint, start from scratch",
    )
    parser.add_argument(
        "--scenario", default="6-bus",
        choices=list(SCENARIOS.keys()),
        help="Grid scenario to simulate (default: 6-bus)",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch web dashboard at http://localhost:8080",
    )
    parser.add_argument(
        "--web-port", type=int, default=8080,
        help="Port for the web dashboard (default: 8080)",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level (WARNING hides library noise from the UI)",
    )

    args = parser.parse_args()

    # Suppress noisy library logs so the Rich UI stays clean
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "conducto_debug.log")],
    )

    run(args)


if __name__ == "__main__":
    main()
