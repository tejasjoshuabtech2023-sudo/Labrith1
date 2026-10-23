"""
Append-only JSONL trace writer and read helpers.

Every agent message, routing decision and proof-of-work record is written
here. No function in this module may modify or delete existing lines.
See Design Doc §6 ("Trace integrity") and §9 (trace.jsonl output).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Trace entry model
# ---------------------------------------------------------------------------
class TraceEntry(BaseModel):
    """One line in the trace JSONL file."""

    run_id: str = Field(..., min_length=1)
    seq: int = Field(..., ge=0, description="Monotonic sequence number within the run.")
    timestamp: str = Field(..., description="ISO 8601 timestamp.")
    agent: str = Field(..., min_length=1, description="Agent that produced this entry.")
    event_type: str = Field(
        ...,
        min_length=1,
        description="Event kind: message, routing, proof, error, etc.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary event data."
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class TraceWriter:
    """Thread-safe, append-only JSONL trace writer.

    Opens the file in append mode.  Every call to ``write`` serialises one
    JSON object per line.  There are intentionally **no** methods to modify or
    delete existing lines — this is a design constraint (Design Doc §6).
    """

    def __init__(self, trace_path: Path, run_id: str) -> None:
        self._path = Path(trace_path)
        self._run_id = run_id
        self._seq = 0
        self._lock = threading.Lock()

        # Ensure the parent directory exists.
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def seq(self) -> int:
        """Current sequence number (next write will use this value)."""
        return self._seq

    def write(
        self,
        agent: str,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> TraceEntry:
        """Append one trace entry.  Returns the written entry."""
        with self._lock:
            entry = TraceEntry(
                run_id=self._run_id,
                seq=self._seq,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent=agent,
                event_type=event_type,
                payload=payload or {},
            )
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
            self._seq += 1
            return entry


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------
def read_trace(trace_path: Path) -> list[TraceEntry]:
    """Read all entries from a trace JSONL file."""
    entries: list[TraceEntry] = []
    path = Path(trace_path)
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(TraceEntry.model_validate_json(line))
            except Exception as exc:
                import logging
                logging.getLogger("frontforge.trace").warning(
                    "Skipping malformed trace line %d in %s: %s", lineno, path, exc
                )
    return entries


def filter_trace(
    entries: list[TraceEntry],
    *,
    agent: Optional[str] = None,
    event_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> list[TraceEntry]:
    """Filter a list of trace entries by agent, event_type and/or run_id."""
    result = entries
    if agent is not None:
        result = [e for e in result if e.agent == agent]
    if event_type is not None:
        result = [e for e in result if e.event_type == event_type]
    if run_id is not None:
        result = [e for e in result if e.run_id == run_id]
    return result
