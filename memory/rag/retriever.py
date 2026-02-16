"""Retriever — queries the vector store at decision time and formats
results into a context block for Nemotron's 1M-token window.

At each agent decision cycle:
  1. Takes the current GridState + alerts
  2. Formulates retrieval queries (congested lines, violated buses, etc.)
  3. Retrieves relevant topology, procedures, and past incidents
  4. Formats them into a context block the agent can reason over
"""

from __future__ import annotations

import logging
from typing import Any

from memory.rag.store import SearchResult, VectorStore
from nemotron_agent.grid_state import GridState

logger = logging.getLogger(__name__)


class GridRetriever:
    """Retrieves relevant grid knowledge for the agent's context window."""

    def __init__(self, store: VectorStore, top_k: int = 10) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve_for_state(self, state: GridState) -> str:
        """Build a RAG context block from the current grid state.

        Automatically formulates queries based on what's happening:
          - Congested lines → retrieve line specs, past incidents
          - Voltage violations → retrieve bus topology, procedures
          - Alerts → retrieve related procedures and incidents
          - General → retrieve operating constraints
        """
        queries = self._build_queries(state)
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for query in queries:
            results = self.store.search(query, top_k=self.top_k)
            for r in results:
                if r.doc_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.doc_id)

        # Sort by relevance
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate and cap total context
        return self._format_context(all_results[:self.top_k * 2])

    def retrieve_query(self, query: str, top_k: int | None = None) -> str:
        """Direct query retrieval — for ad-hoc lookups."""
        results = self.store.search(query, top_k=top_k or self.top_k)
        return self._format_context(results)

    # ── Internal ─────────────────────────────────────────

    def _build_queries(self, state: GridState) -> list[str]:
        """Generate retrieval queries from current grid conditions."""
        queries = []

        # Congested lines
        for line in state.congested_lines:
            queries.append(
                f"transmission line {line.line_id} congestion overload "
                f"{line.from_bus} to {line.to_bus} thermal rating"
            )
            queries.append(
                f"historical incidents line {line.line_id} overload outage"
            )

        # Voltage violations
        for bus in state.voltage_violations:
            direction = "low" if bus.voltage_pu < 0.95 else "high"
            queries.append(
                f"bus {bus.bus_id} voltage {direction} violation operating procedure"
            )

        # Alerts
        for alert in state.alerts:
            queries.append(f"operating procedure for {alert}")

        # Always include general operating constraints
        queries.append("grid operating limits reliability standards NERC")

        # If storage is present, get relevant procedures
        if state.storage:
            queries.append("battery energy storage dispatch procedure SoC limits")

        return queries[:8]  # cap queries to limit latency

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format retrieval results into a text block for the LLM context."""
        if not results:
            return "[RAG CONTEXT: No relevant documents found]"

        parts = ["[RAG CONTEXT — Retrieved grid knowledge]"]

        # Group by category
        by_category: dict[str, list[SearchResult]] = {}
        for r in results:
            cat = r.metadata.get("category", "other")
            by_category.setdefault(cat, []).append(r)

        for cat, items in by_category.items():
            parts.append(f"\n── {cat.upper()} ──")
            for item in items:
                # Truncate long content to fit context budget
                content = item.content[:500]
                parts.append(f"  [{item.doc_id}] {content}")

        return "\n".join(parts)
