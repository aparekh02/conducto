"""Feedback collector — operator approve/reject → training labels.

When an operator approves or rejects an agent recommendation through
the co-pilot UI, this module:
  1. Records the feedback
  2. Attaches it as an Outcome to the corresponding trace
  3. Makes it available for the next SFT fine-tuning round

Approved decisions become high-quality positive labels.
Rejected decisions become negative examples for DPO training.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_DIR = Path(__file__).parent.parent.parent / "nemotron_agent" / "data" / "feedback"


@dataclass
class OperatorFeedback:
    """A single piece of operator feedback on an agent decision."""

    trace_id: str
    operator_id: str
    action: str                    # "approve" | "reject" | "modify"
    timestamp: float = field(default_factory=time.time)
    comment: str = ""
    modified_actions: list[dict] | None = None  # if action == "modify"
    severity_override: str | None = None


class FeedbackCollector:
    """Collects operator feedback and bridges it to the trace buffer."""

    def __init__(
        self,
        feedback_dir: Path | str = DEFAULT_FEEDBACK_DIR,
        trace_buffer: Any = None,
    ) -> None:
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._feedback_file = self.feedback_dir / "feedback.jsonl"
        self._trace_buffer = trace_buffer
        self._feedback_log: list[OperatorFeedback] = []
        logger.info("FeedbackCollector ready  dir=%s", self.feedback_dir)

    def record(self, feedback: OperatorFeedback) -> None:
        """Record operator feedback and optionally attach to trace buffer."""
        self._feedback_log.append(feedback)

        # Persist to JSONL
        with open(self._feedback_file, "a") as f:
            f.write(json.dumps(asdict(feedback)) + "\n")

        # Attach outcome to the trace buffer for training
        if self._trace_buffer is not None:
            from nemotron_agent.trace_buffer import Outcome

            approved = feedback.action == "approve"
            outcome = Outcome(
                operator_approved=approved,
                # Approved decisions get a reward boost, rejected get penalty
                reward=10.0 if approved else -10.0,
            )
            outcome.compute_reward()
            self._trace_buffer.attach_outcome(feedback.trace_id, outcome)

        logger.info(
            "Feedback recorded  trace=%s  action=%s  operator=%s",
            feedback.trace_id, feedback.action, feedback.operator_id,
        )

    def get_stats(self) -> dict[str, int]:
        """Summary stats of operator feedback."""
        total = len(self._feedback_log)
        approved = sum(1 for f in self._feedback_log if f.action == "approve")
        rejected = sum(1 for f in self._feedback_log if f.action == "reject")
        modified = sum(1 for f in self._feedback_log if f.action == "modify")
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "modified": modified,
            "approval_rate": approved / total if total > 0 else 0,
        }

    def get_recent(self, n: int = 20) -> list[dict]:
        return [asdict(f) for f in self._feedback_log[-n:]]
