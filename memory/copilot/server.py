"""Operator co-pilot server — web UI for reviewing agent recommendations.

Operators can:
  - See the current grid state and agent recommendations
  - Understand WHY an action is recommended (the agent's assessment)
  - Approve or reject each recommendation
  - Override with modified actions
  - View decision history and feedback stats

Approvals/rejections feed directly back into the training pipeline
as high-quality labels for the next fine-tuning round.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from memory.copilot.feedback import FeedbackCollector, OperatorFeedback

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ── Request models ───────────────────────────────────────

class FeedbackRequest(BaseModel):
    trace_id: str
    operator_id: str
    action: str                # "approve" | "reject" | "modify"
    comment: str = ""
    modified_actions: list[dict] | None = None


class DecisionView(BaseModel):
    trace_id: str
    timestamp: float
    assessment: str
    risk_level: str
    actions: list[dict]
    grid_summary: str
    latency_ms: float


# ── App factory ──────────────────────────────────────────

def create_app(
    feedback_collector: FeedbackCollector | None = None,
    agent: Any = None,
) -> FastAPI:
    """Create the operator co-pilot FastAPI app."""
    app = FastAPI(title="Conducto Grid Co-Pilot", version="1.0")

    fc = feedback_collector or FeedbackCollector()
    _agent = agent
    _pending_decisions: list[DecisionView] = []

    # ── Dashboard ────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (TEMPLATES_DIR / "dashboard.html").read_text()

    # ── Push a decision for operator review ──────────

    @app.post("/api/decisions")
    def push_decision(decision: DecisionView):
        """Called by the agent loop to push a new decision for review."""
        _pending_decisions.append(decision)
        # Keep only last 100
        while len(_pending_decisions) > 100:
            _pending_decisions.pop(0)
        return {"status": "queued", "pending": len(_pending_decisions)}

    @app.get("/api/decisions")
    def list_decisions(limit: int = 20):
        """List recent decisions awaiting or past review."""
        return {"decisions": [d.model_dump() for d in _pending_decisions[-limit:]]}

    @app.get("/api/decisions/{trace_id}")
    def get_decision(trace_id: str):
        for d in _pending_decisions:
            if d.trace_id == trace_id:
                return d.model_dump()
        raise HTTPException(404, f"Decision {trace_id} not found")

    # ── Operator feedback ────────────────────────────

    @app.post("/api/feedback")
    def submit_feedback(req: FeedbackRequest):
        """Operator approves, rejects, or modifies a decision."""
        feedback = OperatorFeedback(
            trace_id=req.trace_id,
            operator_id=req.operator_id,
            action=req.action,
            comment=req.comment,
            modified_actions=req.modified_actions,
        )
        fc.record(feedback)
        return {
            "status": "recorded",
            "trace_id": req.trace_id,
            "action": req.action,
        }

    @app.get("/api/feedback/stats")
    def feedback_stats():
        return fc.get_stats()

    @app.get("/api/feedback/recent")
    def recent_feedback(limit: int = 20):
        return {"feedback": fc.get_recent(limit)}

    # ── Agent history ────────────────────────────────

    @app.get("/api/agent/history")
    def agent_history(limit: int = 20):
        if _agent is None:
            return {"history": []}
        history = _agent.history[-limit:]
        return {
            "history": [
                {
                    "assessment": d.assessment,
                    "risk_level": d.risk_level,
                    "actions": d.actions,
                    "latency_ms": d.latency_ms,
                    "timestamp": d.timestamp,
                }
                for d in history
            ]
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "pending_decisions": len(_pending_decisions)}

    return app
