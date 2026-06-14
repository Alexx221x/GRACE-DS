"""Lightweight per-episode debug trace logging."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    return _truthy(os.getenv("GRACE_TRACE_ENABLED"), default=False)


def log_executable_code_enabled() -> bool:
    return trace_enabled() and _truthy(
        os.getenv("GRACE_LOG_EXECUTABLE_CODE"), default=True
    )


def log_raw_llm_response_enabled() -> bool:
    return trace_enabled() and _truthy(
        os.getenv("GRACE_LOG_RAW_LLM_RESPONSES"), default=False
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)[:220]


def _base_dir(base_dir: str | os.PathLike[str] | None = None) -> Path | None:
    raw = str(base_dir) if base_dir is not None else os.getenv("GRACE_TRACE_DIR")
    if not raw:
        return None
    return Path(raw)


def _unit_id(unit_id: str | None = None) -> str:
    return unit_id or os.getenv("GRACE_UNIT_ID") or "unknown_unit"


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def trace_event(
    event: str,
    *,
    unit_id: str | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    **fields: Any,
) -> None:
    """Append one JSON event to worker_traces/<unit_id>.jsonl."""
    if base_dir is None and not trace_enabled():
        return
    root = _base_dir(base_dir)
    if root is None:
        return
    uid = _unit_id(unit_id)
    rec = {
        "ts": time.time(),
        "event": event,
        "unit_id": uid,
        "pid": os.getpid(),
        **fields,
    }
    path = root / "worker_traces" / f"{_safe_name(uid)}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        return


def save_text_artifact(
    kind: str,
    text: str,
    *,
    unit_id: str | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    stem: str | None = None,
    suffix: str = ".txt",
) -> tuple[str | None, str]:
    """Save an audit/debug artifact and return (relative_path, sha256)."""
    digest = sha256_text(text or "")
    if base_dir is None and not trace_enabled():
        return None, digest
    root = _base_dir(base_dir)
    if root is None:
        return None, digest
    uid = _unit_id(unit_id)
    safe_stem = _safe_name(stem or f"{_safe_name(uid)}__{digest[:12]}")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    rel = Path(kind) / f"{safe_stem}{suffix}"
    path = root / rel
    try:
        _write_text_atomic(path, text or "")
        return rel.as_posix(), digest
    except Exception:
        return None, digest


def context_stem(
    *,
    unit_id: str | None = None,
    action: str | None = None,
    phase: str | None = None,
    turn: int | None = None,
    trial: int | None = None,
    suffix: str | None = None,
) -> str:
    parts = [_safe_name(_unit_id(unit_id))]
    if phase is not None:
        parts.append(_safe_name(str(phase)))
    if turn is not None:
        parts.append(f"turn{turn}")
    if trial is not None:
        parts.append(f"trial{trial}")
    if action is not None:
        parts.append(_safe_name(str(action)))
    if suffix:
        parts.append(_safe_name(str(suffix)))
    return "__".join(parts)
