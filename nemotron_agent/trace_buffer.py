"""Trace buffer — streams (state, action, outcome) into a persistent JSONL log.

Every agent decision cycle appends a trace.  Later, the training pipeline
samples from this buffer to build SFT and RL datasets.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemotron_agent.grid_state import GridState

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_DIR = Path(__file__).parent / "data" / "traces"


@dataclass
class Outcome:
    """Observed result after an action was applied to the grid."""

    congestion_delta: float = 0.0      # negative = improved
    voltage_violations_delta: int = 0  # negative = improved
    curtailment_mw_delta: float = 0.0  # negative = improved
    cost_delta: float = 0.0            # negative = improved
    operator_approved: bool | None = None  # None = no feedback yet
    reward: float = 0.0                # computed scalar reward

    def compute_reward(self) -> float:
        """Simple scalar reward from multi-objective deltas."""
        r = 0.0
        r -= self.congestion_delta * 2.0
        r -= self.voltage_violations_delta * 5.0
        r -= self.curtailment_mw_delta * 1.0
        r -= self.cost_delta * 0.5
        if self.operator_approved is True:
            r += 10.0
        elif self.operator_approved is False:
            r -= 10.0
        self.reward = r
        return r


@dataclass
class Trace:
    """One decision cycle: state → action → outcome."""

    trace_id: str
    timestamp: float
    state: dict[str, Any]
    messages: list[dict[str, str]]  # the prompt sent to Nemotron
    response: dict[str, Any]        # parsed agent decision
    raw_response: str
    outcome: dict[str, Any] | None = None
    reward: float = 0.0
    latency_ms: float = 0.0


class TraceBuffer:
    """Append-only JSONL buffer on disk with in-memory index for sampling."""

    def __init__(self, buffer_dir: Path | str = DEFAULT_BUFFER_DIR) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self._traces_file = self.buffer_dir / "traces.jsonl"
        self._index: list[dict[str, Any]] = []  # lightweight index
        self._load_index()
        logger.info(
            "TraceBuffer ready  dir=%s  existing_traces=%d",
            self.buffer_dir,
            len(self._index),
        )

    def _load_index(self) -> None:
        """Rebuild in-memory index from existing JSONL."""
        if not self._traces_file.exists():
            return
        with open(self._traces_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self._index.append({
                        "trace_id": rec["trace_id"],
                        "timestamp": rec["timestamp"],
                        "reward": rec.get("reward", 0.0),
                        "has_outcome": rec.get("outcome") is not None,
                        "operator_approved": (
                            rec.get("outcome", {}) or {}
                        ).get("operator_approved"),
                    })

    # ── Write ────────────────────────────────────────────

    def record(
        self,
        state: GridState,
        messages: list[dict[str, str]],
        decision: Any,
        outcome: Outcome | None = None,
    ) -> str:
        """Append a trace. Returns the trace_id."""
        trace_id = f"tr-{int(time.time() * 1000)}-{random.randint(0, 9999):04d}"

        outcome_dict = None
        reward = 0.0
        if outcome is not None:
            outcome.compute_reward()
            reward = outcome.reward
            outcome_dict = asdict(outcome)

        trace = Trace(
            trace_id=trace_id,
            timestamp=time.time(),
            state=state.to_dict(),
            messages=messages,
            response={
                "assessment": decision.assessment,
                "risk_level": decision.risk_level,
                "actions": decision.actions,
            },
            raw_response=decision.raw_response,
            outcome=outcome_dict,
            reward=reward,
            latency_ms=decision.latency_ms,
        )

        with open(self._traces_file, "a") as f:
            f.write(json.dumps(asdict(trace)) + "\n")

        self._index.append({
            "trace_id": trace_id,
            "timestamp": trace.timestamp,
            "reward": reward,
            "has_outcome": outcome is not None,
            "operator_approved": (
                outcome.operator_approved if outcome else None
            ),
        })

        logger.debug("Recorded trace %s  reward=%.2f", trace_id, reward)
        return trace_id

    def attach_outcome(self, trace_id: str, outcome: Outcome) -> None:
        """Attach an outcome to a previously recorded trace (backfill)."""
        outcome.compute_reward()
        lines = []
        with open(self._traces_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec["trace_id"] == trace_id:
                    rec["outcome"] = asdict(outcome)
                    rec["reward"] = outcome.reward
                lines.append(json.dumps(rec))

        with open(self._traces_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        for entry in self._index:
            if entry["trace_id"] == trace_id:
                entry["reward"] = outcome.reward
                entry["has_outcome"] = True
                entry["operator_approved"] = outcome.operator_approved
                break

        logger.debug("Attached outcome to %s  reward=%.2f", trace_id, outcome.reward)

    # ── Read / Sample ────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._index)

    def sample_traces(
        self,
        n: int,
        *,
        only_with_outcome: bool = False,
        only_approved: bool = False,
        min_reward: float | None = None,
    ) -> list[dict[str, Any]]:
        """Sample n full trace records matching the filters."""
        candidates = self._index
        if only_with_outcome:
            candidates = [c for c in candidates if c["has_outcome"]]
        if only_approved:
            candidates = [c for c in candidates if c["operator_approved"] is True]
        if min_reward is not None:
            candidates = [c for c in candidates if c["reward"] >= min_reward]

        if not candidates:
            return []

        selected_ids = {
            e["trace_id"]
            for e in random.sample(candidates, min(n, len(candidates)))
        }

        traces = []
        with open(self._traces_file) as f:
            for line in f:
                rec = json.loads(line)
                if rec["trace_id"] in selected_ids:
                    traces.append(rec)
        return traces

    def export_jsonl(self, path: Path | str, **filter_kwargs: Any) -> int:
        """Export filtered traces to a new JSONL file. Returns count."""
        traces = self.sample_traces(self.size, **filter_kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        logger.info("Exported %d traces to %s", len(traces), path)
        return len(traces)
