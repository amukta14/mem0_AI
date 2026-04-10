from __future__ import annotations

import os
from pathlib import Path


class UnsafePathError(ValueError):
    pass


def repo_root() -> Path:
    # repo root is two levels above: src/voice_agent -> repo
    return Path(__file__).resolve().parents[2]


def output_root() -> Path:
    return repo_root() / "output"


def _safe_output_path(user_path: str) -> Path:
    if not user_path or user_path.strip() == "":
        raise ValueError("path is required")

    # Normalize path and force it under output/
    p = Path(user_path.strip())
    if p.is_absolute():
        # Convert absolute to relative name only, to avoid surprises.
        p = Path(p.name)

    resolved = (output_root() / p).resolve()
    out = output_root().resolve()

    try:
        resolved.relative_to(out)
    except Exception as e:
        raise UnsafePathError(f"Refusing to write outside output/: {user_path}") from e

    return resolved


def create_file(user_path: str, overwrite: bool = False) -> str:
    target = _safe_output_path(user_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return f"File already exists: {target.relative_to(output_root())}"
    target.write_text("", encoding="utf-8")
    return f"Created file: {target.relative_to(output_root())}"


def write_text_file(user_path: str, content: str, overwrite: bool = True) -> str:
    target = _safe_output_path(user_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return f"Skipped (exists): {target.relative_to(output_root())}"
    target.write_text(content or "", encoding="utf-8")
    return f"Wrote {len((content or '').encode('utf-8'))} bytes to: {target.relative_to(output_root())}"


def list_output_tree(max_entries: int = 200) -> list[str]:
    out = output_root()
    if not out.exists():
        return []
    rows: list[str] = []
    for root, _dirs, files in os.walk(out):
        for f in files:
            rel = (Path(root) / f).relative_to(out)
            rows.append(str(rel))
            if len(rows) >= max_entries:
                return rows
    return rows

