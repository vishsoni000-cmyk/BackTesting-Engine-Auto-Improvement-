"""Persistent error memory with repeat detection and confirmed resolution notes.

The module stores diagnostics and safe, human-confirmed resolutions. It never executes
arbitrary repair code or rewrites project source files.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
)


def _redact(text: str) -> str:
    """Remove common credential-like values before writing diagnostics to disk."""
    result = text
    for pattern in _SECRET_PATTERNS:
        if "api" in pattern.pattern.lower() or "secret" in pattern.pattern.lower():
            result = pattern.sub(r"\1=[REDACTED]", result)
        else:
            result = pattern.sub("Bearer [REDACTED]", result)
    return result


def error_fingerprint(error_type: str, message: str) -> str:
    """Return a stable fingerprint after normalizing volatile numbers and whitespace."""
    normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "<address>", message)
    normalized = re.sub(r"\b\d+\b", "<number>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    payload = f"{error_type.strip()}::{normalized}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


class ErrorMemory:
    """Thread-safe JSON registry of recurring errors and their confirmed resolutions."""

    def __init__(self, path: str | Path = "data/error_memory.json") -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._records = {}
                return
            try:
                parsed = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("Error memory must contain a JSON object")
                self._records = {
                    str(key): value for key, value in parsed.items() if isinstance(value, dict)
                }
            except (OSError, json.JSONDecodeError, ValueError):
                # Keep the damaged evidence as a backup; do not silently overwrite it.
                backup = self.path.with_suffix(self.path.suffix + ".corrupt")
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
                self._records = {}

    def record(self, error: BaseException, context: str = "") -> dict[str, Any]:
        """Record an exception, incrementing the count when its fingerprint already exists."""
        error_type = type(error).__name__
        message = _redact(str(error))[:2000]
        fingerprint = error_fingerprint(error_type, message)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            record = self._records.setdefault(
                fingerprint,
                {
                    "fingerprint": fingerprint,
                    "error_type": error_type,
                    "message": message,
                    "context": _redact(context)[:1000],
                    "occurrences": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "resolution": None,
                    "regression_test": None,
                },
            )
            record["occurrences"] = int(record.get("occurrences", 0)) + 1
            record["last_seen"] = now
            if context and not record.get("context"):
                record["context"] = _redact(context)[:1000]
            self._save()
            return dict(record)

    def resolve(
        self, fingerprint: str, resolution: str, regression_test: str | None = None
    ) -> bool:
        """Store a confirmed resolution. Return False if the fingerprint is unknown."""
        note = _redact(resolution).strip()
        if not note:
            raise ValueError("resolution must not be empty")
        with self._lock:
            record = self._records.get(fingerprint)
            if record is None:
                return False
            record["resolution"] = note[:2000]
            record["regression_test"] = regression_test.strip()[:500] if regression_test else None
            record["resolved_at"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return True

    def find(self, fingerprint: str) -> dict[str, Any] | None:
        """Return a shallow copy of the matching error record, if any."""
        with self._lock:
            record = self._records.get(fingerprint)
            return dict(record) if record is not None else None

    def known_resolution(self, fingerprint: str) -> str | None:
        """Return a previously confirmed resolution, if one exists."""
        record = self.find(fingerprint)
        if record is None:
            return None
        resolution = record.get("resolution")
        return str(resolution) if resolution else None

    def all_records(self) -> list[dict[str, Any]]:
        """Return copies of all stored records."""
        with self._lock:
            return [dict(record) for record in self._records.values()]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._records, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
