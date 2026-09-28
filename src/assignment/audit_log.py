"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, tuple[float, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record the question and start time for one request."""
        key = request_id or user_id
        self._open[key] = (time.monotonic(), utc_now_iso())
        self.logs.append({"request_id": key, "user_id": user_id, "input": text,
                          "started_at": self._open[key][1]})

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching request record with its decision and latency."""
        key = request_id or user_id
        started = self._open.pop(key, None)
        record = next((row for row in reversed(self.logs)
                       if row["request_id"] == key and "output" not in row), None)
        if record is None:
            record = {"request_id": key, "user_id": user_id, "input": None,
                      "started_at": utc_now_iso()}
            self.logs.append(record)
        record.update({"output": text, "blocked": blocked, "layer": layer,
                       "completed_at": utc_now_iso(),
                       "latency_ms": round((time.monotonic() - started[0]) * 1000, 2)
                       if started else None})

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
