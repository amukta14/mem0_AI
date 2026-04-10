from __future__ import annotations

import json
from typing import Any

import requests

from src.voice_agent.types import PlannedAction


PLANNER_SYSTEM = """You are an intent classifier and action planner for a local voice-controlled agent.
You MUST output a single JSON object and nothing else.

The agent supports these intents:
- create_file: create an empty file under output/
- write_code: write code content to a file under output/
- summarize_text: summarize user-provided text
- general_chat: reply conversationally

Rules:
- Support compound commands by returning multiple actions in "actions".
- Any file path must be relative and MUST NOT contain '..' or start with '/'. Keep it short.
- Prefer markdown-friendly filenames for summaries (e.g. summary.md) and code filenames with extensions.
- If the user asks to write code but doesn't name a file, choose a reasonable default filename.

JSON schema:
{
  "actions": [
    {
      "intent": "create_file" | "write_code" | "summarize_text" | "general_chat",
      "path": "relative/path.ext" (only for file intents),
      "language": "python|js|ts|..." (optional, for write_code),
      "content": "file contents" (only for write_code),
      "input_text": "text to summarize" (only for summarize_text; if not provided, summarize the transcript),
      "rationale": "short reason"
    }
  ],
  "final_response": "string response to show in UI"
}
"""


def _post_ollama_chat(host: str, model: str, messages: list[dict[str, str]]) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False, "format": "json"}
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    # Ollama returns: { message: { content: "..." } }
    return data.get("message", {}).get("content", "")


def plan_actions(transcript: str, *, ollama_host: str, ollama_model: str) -> tuple[list[PlannedAction], str, dict[str, Any]]:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": transcript},
    ]
    raw = _post_ollama_chat(ollama_host, ollama_model, messages)
    debug = {"raw_planner_output": raw}

    try:
        obj = json.loads(raw)
    except Exception:
        # Try to salvage if model wrapped JSON in text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
        else:
            raise RuntimeError("Planner did not return valid JSON. Try again or switch model.")

    actions_in = obj.get("actions", [])
    final_response = obj.get("final_response", "") or ""

    actions: list[PlannedAction] = []
    for a in actions_in:
        if not isinstance(a, dict):
            continue
        actions.append(
            PlannedAction(
                intent=a.get("intent"),
                path=a.get("path"),
                content=a.get("content"),
                language=a.get("language"),
                input_text=a.get("input_text"),
                rationale=a.get("rationale"),
            )
        )

    if not actions:
        # Minimal fallback
        actions = [PlannedAction(intent="general_chat", input_text=transcript, rationale="Fallback: no actions parsed")]
        final_response = final_response or "I couldn't parse a plan, so I'm responding conversationally."

    return actions, final_response, debug


def chat_reply(transcript: str, *, ollama_host: str, ollama_model: str, context: list[dict[str, str]] | None = None) -> str:
    messages = [{"role": "system", "content": "You are a helpful local assistant. Keep replies concise."}]
    if context:
        messages.extend(context[-12:])
    messages.append({"role": "user", "content": transcript})
    return _post_ollama_chat(ollama_host, ollama_model, messages).strip()

