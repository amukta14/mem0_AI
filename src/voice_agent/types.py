from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


Intent = Literal["create_file", "write_code", "summarize_text", "general_chat"]


@dataclass
class PlannedAction:
    intent: Intent
    # For file operations (restricted to output/)
    path: Optional[str] = None
    # For write_code
    content: Optional[str] = None
    language: Optional[str] = None
    # For summarization/chat
    input_text: Optional[str] = None
    # Free-form LLM reasoning is not required; keep minimal but useful for UI.
    rationale: Optional[str] = None


@dataclass
class ExecutionStep:
    label: str
    status: Literal["ok", "skipped", "error"]
    details: dict[str, Any] | None = None


@dataclass
class AgentRunResult:
    transcript: str
    actions: list[PlannedAction]
    steps: list[ExecutionStep] = field(default_factory=list)
    final_output: str = ""
    needs_confirmation: bool = False
    approved: bool = False
    # If needs_confirmation=True, we keep actions but do not execute file ops until approved.
    pending_action_indexes: list[int] = field(default_factory=list)

