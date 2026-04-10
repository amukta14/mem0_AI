from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.voice_agent.types import AgentRunResult


def _history_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "history.jsonl"


def append_history(result: AgentRunResult) -> None:
    p = _history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = asdict(result)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_history(limit: int | None = None) -> list[dict[str, Any]]:
    p = _history_path()
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit is not None:
        return rows[-limit:]
    return rows

