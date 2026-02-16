"""Nemotron Grid Pressure Autopilot — core agent loop.

Each cycle:
  1. Receive a GridState snapshot (from SCADA / sim / demo data).
  2. Build a prompt with the system persona + state context.
  3. Ask Nemotron (via NIM cloud API) for recommended actions.
  4. Return structured actions for the operator / simulator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nemotron_agent.config import Config
from nemotron_agent.grid_state import GridState
from nemotron_agent.nim_client import NIMClient
from nemotron_agent.trace_buffer import Outcome, TraceBuffer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Conducto Grid Pressure Autopilot, a real-time power-grid control
agent built on NVIDIA Nemotron.  Your job is to keep the grid stable, reduce
congestion, minimise curtailment of renewables, and lower operating cost.

Given the current grid state, propose a set of control actions.  Each action
must be one of:
  • REDISPATCH   — change a generator's output (unit_id, target_mw)
  • STORAGE      — charge or discharge a battery (unit_id, setpoint_mw)
  • LOAD_SHAPE   — adjust flexible load / EV charging (bus_id, delta_mw)
  • TOPOLOGY     — open or close a switch (line_id, action: open|close)

Respond ONLY with a JSON object:
{
  "assessment": "<one-sentence grid situation summary>",
  "risk_level": "low" | "medium" | "high" | "critical",
  "actions": [
    {"type": "<ACTION_TYPE>", "target": "<id>", "params": {<key-value>}, "reason": "<why>"}
  ]
}
Do not include any text outside the JSON object.
"""


@dataclass
class AgentDecision:
    """Structured output from a single agent decision cycle."""

    assessment: str
    risk_level: str
    actions: list[dict[str, Any]]
    raw_response: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class NemotronGridAgent:
    """The main agent.  Stateless per-cycle; state lives in GridState."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.nim = NIMClient(self.cfg.nvidia)
        self.traces = TraceBuffer()
        self._history: list[AgentDecision] = []
        self._outcomes: list[Outcome | None] = []  # parallel to _history
        self._last_trace_id: str | None = None
        logger.info("NemotronGridAgent initialised")

    # ── Public API ───────────────────────────────────────

    def decide(
        self,
        state: GridState,
        outcome: Outcome | None = None,
    ) -> AgentDecision:
        """Run one decision cycle: state → Nemotron → structured actions.

        If an Outcome from the *previous* cycle is provided, it is attached
        to the prior trace (backfill) so the training pipeline can use it.
        """
        # Backfill outcome onto the previous trace AND store for context
        if outcome is not None and self._last_trace_id is not None:
            self.traces.attach_outcome(self._last_trace_id, outcome)
        if self._outcomes and outcome is not None:
            self._outcomes[-1] = outcome  # attach to the last decision
        elif outcome is not None and not self._outcomes:
            pass  # first cycle, no prior decision to attach to

        messages = self._build_messages(state, outcome)

        t0 = time.perf_counter()
        raw = self.nim.chat(
            messages,
            temperature=self.cfg.agent.temperature,
            max_tokens=self.cfg.agent.max_output_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        parsed = self._parse_response(raw)
        decision = AgentDecision(
            assessment=parsed.get("assessment", ""),
            risk_level=parsed.get("risk_level", "unknown"),
            actions=parsed.get("actions", []),
            raw_response=raw,
            latency_ms=latency_ms,
        )
        self._history.append(decision)
        self._outcomes.append(None)  # placeholder, filled in next cycle

        # Record trace for training pipeline
        self._last_trace_id = self.traces.record(state, messages, decision)

        logger.info(
            "Decision  risk=%s  actions=%d  latency=%.0fms  trace=%s",
            decision.risk_level,
            len(decision.actions),
            decision.latency_ms,
            self._last_trace_id,
        )
        return decision

    @property
    def history(self) -> list[AgentDecision]:
        return list(self._history)

    # ── Internal ─────────────────────────────────────────

    def _build_messages(
        self, state: GridState, latest_outcome: Outcome | None = None,
    ) -> list[dict[str, str]]:
        context = state.to_context_block()

        # Include ALL bus data so the LLM knows individual gen/load per bus
        bus_table = "\n\nBUS DETAIL:"
        for b in state.buses:
            net = b.generation_mw - b.load_mw
            bus_table += (
                f"\n  {b.bus_id}  V={b.voltage_pu:.3f}pu  "
                f"load={b.load_mw:.0f}MW  gen={b.generation_mw:.0f}MW  "
                f"net={net:+.0f}MW"
            )

        # Include last 3 decisions WITH their outcomes (reward feedback)
        history_block = ""
        n = len(self._history)
        for i in range(max(0, n - 3), n):
            d = self._history[i]
            o = self._outcomes[i]
            actions_summary = ", ".join(
                f"{a.get('type','?')}→{a.get('target','?')}"
                for a in d.actions[:4]
            ) or "none"
            outcome_str = ""
            if o is not None:
                outcome_str = (
                    f"  RESULT: reward={o.reward:+.1f} "
                    f"congestion_delta={o.congestion_delta:+d} "
                    f"curtailment_delta={o.curtailment_mw_delta:+.1f}MW"
                )
                if o.reward < -10:
                    outcome_str += " ** BAD — avoid repeating these actions **"
                elif o.reward > 0:
                    outcome_str += " (good — similar approach may help)"
            history_block += (
                f"\n[CYCLE {i+1}  risk={d.risk_level}  "
                f"actions: {actions_summary}]: {d.assessment}"
                f"{outcome_str}"
            )

        # Build the user message with full context
        user_content = context + bus_table
        if history_block:
            user_content += "\n\nPRIOR DECISIONS AND OUTCOMES:" + history_block
        if latest_outcome is not None:
            user_content += (
                f"\n\nLAST CYCLE OUTCOME: reward={latest_outcome.reward:+.1f}"
            )
            if latest_outcome.reward < -50:
                user_content += (
                    "\nWARNING: Very negative reward. Your previous actions "
                    "made the grid WORSE. Try a different strategy."
                )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse Nemotron response as JSON")
            return {
                "assessment": cleaned[:200],
                "risk_level": "unknown",
                "actions": [],
            }
