"""Document indexer — loads grid knowledge into the vector store.

Indexes three categories of documents:
  1. Grid topology and equipment models
  2. Operating procedures, NERC/ISO rules, and constraints
  3. Historical outage and incident reports

Each document is chunked for efficient retrieval and tagged with
metadata (category, date, equipment_id, etc.) for filtering.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from memory.rag.store import Document, VectorStore

logger = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _doc_id(content: str, prefix: str) -> str:
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}-{h}"


class GridDocIndexer:
    """Indexes grid knowledge documents into the vector store."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store

    # ── 1. Grid topology ─────────────────────────────────

    def index_topology(self, topology_data: dict[str, Any]) -> int:
        """Index structured grid topology (buses, lines, transformers).

        Expects a dict like:
        {
            "buses": [{"bus_id": ..., "voltage_kv": ..., "zone": ..., ...}],
            "lines": [{"line_id": ..., "from": ..., "to": ..., "rating_mw": ..., ...}],
            "transformers": [...],
        }
        """
        docs = []
        for bus in topology_data.get("buses", []):
            text = (
                f"Bus {bus['bus_id']}: {bus.get('name', '')}. "
                f"Nominal voltage: {bus.get('voltage_kv', 'N/A')} kV. "
                f"Zone: {bus.get('zone', 'N/A')}. "
                f"Type: {bus.get('type', 'load')}. "
                f"Connected equipment: {', '.join(bus.get('equipment', []))}."
            )
            docs.append(Document(
                doc_id=_doc_id(text, "topo-bus"),
                content=text,
                metadata={"category": "topology", "subtype": "bus", "bus_id": bus["bus_id"]},
            ))

        for line in topology_data.get("lines", []):
            text = (
                f"Line {line['line_id']}: connects {line['from']} to {line['to']}. "
                f"Thermal rating: {line.get('rating_mw', 'N/A')} MW. "
                f"Voltage: {line.get('voltage_kv', 'N/A')} kV. "
                f"Length: {line.get('length_km', 'N/A')} km. "
                f"Impedance: R={line.get('r_pu', 'N/A')}, X={line.get('x_pu', 'N/A')} pu."
            )
            docs.append(Document(
                doc_id=_doc_id(text, "topo-line"),
                content=text,
                metadata={"category": "topology", "subtype": "line", "line_id": line["line_id"]},
            ))

        for xfmr in topology_data.get("transformers", []):
            text = (
                f"Transformer {xfmr['xfmr_id']}: {xfmr.get('from_kv', '?')} kV / "
                f"{xfmr.get('to_kv', '?')} kV. "
                f"Rating: {xfmr.get('rating_mva', 'N/A')} MVA. "
                f"Tap range: {xfmr.get('tap_min', 'N/A')}–{xfmr.get('tap_max', 'N/A')}."
            )
            docs.append(Document(
                doc_id=_doc_id(text, "topo-xfmr"),
                content=text,
                metadata={"category": "topology", "subtype": "transformer"},
            ))

        return self.store.add(docs) if docs else 0

    # ── 2. Operating procedures and rules ────────────────

    def index_procedures(self, documents: list[dict[str, str]]) -> int:
        """Index operating procedures and regulatory constraints.

        Expects a list of:
        [{"title": "...", "body": "...", "source": "NERC TPL-001", ...}]
        """
        all_docs = []
        for doc in documents:
            chunks = _chunk_text(doc["body"])
            for i, chunk in enumerate(chunks):
                text = f"[{doc['title']}] {chunk}"
                all_docs.append(Document(
                    doc_id=_doc_id(text, "proc"),
                    content=text,
                    metadata={
                        "category": "procedure",
                        "title": doc["title"],
                        "source": doc.get("source", ""),
                        "chunk_index": i,
                    },
                ))
        return self.store.add(all_docs) if all_docs else 0

    # ── 3. Historical incidents ──────────────────────────

    def index_incidents(self, incidents: list[dict[str, Any]]) -> int:
        """Index historical outage and incident reports.

        Expects a list of:
        [{
            "incident_id": "...",
            "date": "2024-08-15",
            "severity": "major",
            "description": "...",
            "root_cause": "...",
            "equipment_affected": ["LN-01", "BUS-002"],
            "resolution": "...",
            "lessons_learned": "...",
        }]
        """
        docs = []
        for inc in incidents:
            text = (
                f"Incident {inc['incident_id']} ({inc.get('date', 'unknown')}): "
                f"Severity: {inc.get('severity', 'N/A')}. "
                f"{inc['description']} "
                f"Root cause: {inc.get('root_cause', 'N/A')}. "
                f"Equipment affected: {', '.join(inc.get('equipment_affected', []))}. "
                f"Resolution: {inc.get('resolution', 'N/A')}. "
                f"Lessons learned: {inc.get('lessons_learned', 'N/A')}."
            )
            docs.append(Document(
                doc_id=_doc_id(text, "incident"),
                content=text,
                metadata={
                    "category": "incident",
                    "incident_id": inc["incident_id"],
                    "date": inc.get("date", ""),
                    "severity": inc.get("severity", ""),
                    "equipment": inc.get("equipment_affected", []),
                },
            ))
        return self.store.add(docs) if docs else 0

    # ── Bulk load from files ─────────────────────────────

    def index_from_directory(self, docs_dir: Path | str) -> int:
        """Load and index all JSON files from a directory.

        Expected file naming:
          topology.json    → grid topology
          procedures/*.json → operating procedures
          incidents/*.json  → historical incidents
        """
        docs_dir = Path(docs_dir)
        total = 0

        topo_file = docs_dir / "topology.json"
        if topo_file.exists():
            with open(topo_file) as f:
                total += self.index_topology(json.load(f))

        proc_dir = docs_dir / "procedures"
        if proc_dir.is_dir():
            for fp in proc_dir.glob("*.json"):
                with open(fp) as f:
                    total += self.index_procedures(json.load(f))

        inc_dir = docs_dir / "incidents"
        if inc_dir.is_dir():
            for fp in inc_dir.glob("*.json"):
                with open(fp) as f:
                    total += self.index_incidents(json.load(f))

        logger.info("Indexed %d documents from %s", total, docs_dir)
        return total
