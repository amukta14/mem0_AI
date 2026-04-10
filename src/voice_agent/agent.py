from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.voice_agent.fs_tools import UnsafePathError, create_file, list_output_tree, write_text_file
from src.voice_agent.llm_planner import chat_reply, plan_actions
from src.voice_agent.stt import transcribe_wav_bytes
from src.voice_agent.types import AgentRunResult, ExecutionStep, PlannedAction


def _looks_unsafe(path: str | None) -> bool:
    if not path:
        return True
    p = path.strip()
    return p.startswith("/") or ".." in p.replace("\\", "/").split("/")


class Agent:
    def run_text(
        self,
        *,
        transcript: str,
        ollama_host: str,
        ollama_model: str,
        require_confirmation: bool,
    ) -> AgentRunResult:
        steps: list[ExecutionStep] = []

        transcript = (transcript or "").strip()
        if not transcript:
            return AgentRunResult(
                transcript="",
                actions=[],
                steps=[ExecutionStep(label="Text input", status="error", details={"error": "Empty input"})],
                final_output="Please enter a command to run.",
            )

        steps.append(ExecutionStep(label="Text input", status="ok", details={"chars": len(transcript)}))

        # Plan
        try:
            actions, final_response, debug = plan_actions(transcript, ollama_host=ollama_host, ollama_model=ollama_model)
            steps.append(
                ExecutionStep(
                    label="Intent planning (LLM)",
                    status="ok",
                    details={"actions": [asdict(a) for a in actions], **debug},
                )
            )
        except Exception as e:
            # fallback to chat mode
            try:
                reply = chat_reply(transcript, ollama_host=ollama_host, ollama_model=ollama_model)
            except Exception as e2:
                reply = f"LLM planner failed ({e}); chat fallback also failed ({e2})."
            return AgentRunResult(
                transcript=transcript,
                actions=[PlannedAction(intent="general_chat", input_text=transcript, rationale="Planner failed")],
                steps=steps
                + [ExecutionStep(label="Intent planning (LLM)", status="error", details={"error": str(e)})],
                final_output=reply,
            )

        result = AgentRunResult(transcript=transcript, actions=actions, steps=steps, final_output=final_response)

        # Determine if confirmation needed
        pending_idxs: list[int] = []
        for i, a in enumerate(actions):
            if a.intent in {"create_file", "write_code"}:
                pending_idxs.append(i)
        result.pending_action_indexes = pending_idxs
        result.needs_confirmation = bool(pending_idxs) and require_confirmation

        if result.needs_confirmation:
            result.steps.append(
                ExecutionStep(
                    label="Human-in-the-loop gate",
                    status="skipped",
                    details={
                        "message": "Confirmation required before file operations.",
                        "pending_action_indexes": pending_idxs,
                    },
                )
            )
            result.steps.append(
                ExecutionStep(
                    label="Output folder snapshot (preview)",
                    status="ok",
                    details={"output_files": list_output_tree()},
                )
            )
            return result

        executed = self._execute_actions(actions, transcript=transcript, ollama_host=ollama_host, ollama_model=ollama_model)
        executed.final_output = final_response or executed.final_output
        executed.approved = True
        return executed

    def run(
        self,
        *,
        wav_bytes: bytes,
        stt_sample_rate: int | None,
        whisper_model: str,
        ollama_host: str,
        ollama_model: str,
        require_confirmation: bool,
    ) -> AgentRunResult:
        steps: list[ExecutionStep] = []

        # STT
        try:
            transcript = transcribe_wav_bytes(wav_bytes, model_name=whisper_model, sample_rate=stt_sample_rate)
            steps.append(ExecutionStep(label="Speech-to-text", status="ok", details={"transcript": transcript}))
        except Exception as e:
            return AgentRunResult(
                transcript="",
                actions=[],
                steps=[ExecutionStep(label="Speech-to-text", status="error", details={"error": str(e)})],
                final_output="STT failed. See error details above.",
            )

        # Plan
        try:
            actions, final_response, debug = plan_actions(transcript, ollama_host=ollama_host, ollama_model=ollama_model)
            steps.append(
                ExecutionStep(
                    label="Intent planning (LLM)",
                    status="ok",
                    details={"actions": [asdict(a) for a in actions], **debug},
                )
            )
        except Exception as e:
            # fallback to chat mode
            try:
                reply = chat_reply(transcript, ollama_host=ollama_host, ollama_model=ollama_model)
            except Exception as e2:
                reply = f"LLM planner failed ({e}); chat fallback also failed ({e2})."
            return AgentRunResult(
                transcript=transcript,
                actions=[PlannedAction(intent="general_chat", input_text=transcript, rationale="Planner failed")],
                steps=steps
                + [ExecutionStep(label="Intent planning (LLM)", status="error", details={"error": str(e)})],
                final_output=reply,
            )

        result = AgentRunResult(transcript=transcript, actions=actions, steps=steps, final_output=final_response)

        # Determine if confirmation needed
        pending_idxs: list[int] = []
        for i, a in enumerate(actions):
            if a.intent in {"create_file", "write_code"}:
                pending_idxs.append(i)
        result.pending_action_indexes = pending_idxs
        result.needs_confirmation = bool(pending_idxs) and require_confirmation

        # Execute immediately if allowed
        if result.needs_confirmation:
            result.steps.append(
                ExecutionStep(
                    label="Human-in-the-loop gate",
                    status="skipped",
                    details={
                        "message": "Confirmation required before file operations.",
                        "pending_action_indexes": pending_idxs,
                    },
                )
            )
            # Provide a helpful preview
            result.steps.append(
                ExecutionStep(
                    label="Output folder snapshot (preview)",
                    status="ok",
                    details={"output_files": list_output_tree()},
                )
            )
            return result

        executed = self._execute_actions(actions, transcript=transcript, ollama_host=ollama_host, ollama_model=ollama_model)
        executed.final_output = final_response or executed.final_output
        executed.approved = True
        return executed

    def execute_approved(self, prior: AgentRunResult) -> AgentRunResult:
        # Re-run only the pending actions, preserving transcript and action plan
        actions = prior.actions
        idxs = prior.pending_action_indexes or list(range(len(actions)))
        sub = [actions[i] for i in idxs if 0 <= i < len(actions)]

        executed = self._execute_actions(
            sub,
            transcript=prior.transcript,
            ollama_host="",  # not needed unless general_chat fallback
            ollama_model="",
        )

        # Merge execution steps with prior steps for full traceability
        merged_steps = list(prior.steps)
        merged_steps.append(
            ExecutionStep(label="Human-in-the-loop gate", status="ok", details={"approved": True, "executed": idxs})
        )
        merged_steps.extend(executed.steps)

        return AgentRunResult(
            transcript=prior.transcript,
            actions=prior.actions,
            steps=merged_steps,
            final_output=executed.final_output or prior.final_output,
            needs_confirmation=False,
            approved=True,
            pending_action_indexes=[],
        )

    def _execute_actions(
        self,
        actions: list[PlannedAction],
        *,
        transcript: str,
        ollama_host: str,
        ollama_model: str,
    ) -> AgentRunResult:
        steps: list[ExecutionStep] = []
        outputs: list[str] = []

        for a in actions:
            if a.intent == "create_file":
                if _looks_unsafe(a.path):
                    steps.append(
                        ExecutionStep(label="Create file", status="error", details={"error": "Unsafe or missing path"})
                    )
                    continue
                try:
                    msg = create_file(a.path)
                    outputs.append(msg)
                    steps.append(ExecutionStep(label="Create file", status="ok", details={"path": a.path, "result": msg}))
                except (UnsafePathError, Exception) as e:
                    steps.append(ExecutionStep(label="Create file", status="error", details={"path": a.path, "error": str(e)}))

            elif a.intent == "write_code":
                path = a.path or "generated.py"
                if _looks_unsafe(path):
                    steps.append(
                        ExecutionStep(label="Write code", status="error", details={"error": "Unsafe path"})
                    )
                    continue
                content = a.content
                if not content:
                    steps.append(
                        ExecutionStep(label="Write code", status="error", details={"path": path, "error": "Missing content"})
                    )
                    continue
                try:
                    msg = write_text_file(path, content, overwrite=True)
                    outputs.append(msg)
                    steps.append(
                        ExecutionStep(
                            label="Write code",
                            status="ok",
                            details={"path": path, "language": a.language, "bytes": len(content.encode("utf-8")), "result": msg},
                        )
                    )
                except (UnsafePathError, Exception) as e:
                    steps.append(ExecutionStep(label="Write code", status="error", details={"path": path, "error": str(e)}))

            elif a.intent == "summarize_text":
                # Simple deterministic summary (no extra model required). If input_text missing, summarize transcript.
                text = (a.input_text or transcript or "").strip()
                if not text:
                    steps.append(ExecutionStep(label="Summarize", status="error", details={"error": "No text to summarize"}))
                    continue
                summary = _cheap_summary(text)
                outputs.append(summary)
                steps.append(ExecutionStep(label="Summarize", status="ok", details={"chars_in": len(text), "chars_out": len(summary)}))

                # Optional: user can pair this with write_code/create_file via compound planning.

            elif a.intent == "general_chat":
                if ollama_host and ollama_model:
                    try:
                        reply = chat_reply(transcript, ollama_host=ollama_host, ollama_model=ollama_model)
                    except Exception as e:
                        reply = f"(Chat failed) {e}"
                else:
                    reply = "Approved actions executed."
                outputs.append(reply)
                steps.append(ExecutionStep(label="Chat", status="ok", details={"reply": reply}))
            else:
                steps.append(ExecutionStep(label="Unknown intent", status="error", details={"intent": a.intent}))

        steps.append(ExecutionStep(label="Output folder snapshot", status="ok", details={"output_files": list_output_tree()}))
        final = "\n\n".join([o for o in outputs if o]).strip() or "Done."
        return AgentRunResult(transcript=transcript, actions=actions, steps=steps, final_output=final)


def _cheap_summary(text: str, max_bullets: int = 6) -> str:
    """
    Lightweight summarizer: extracts key sentences and compresses.
    This keeps the pipeline local even without an LLM.
    """
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= 280:
        return cleaned

    # Sentence-ish split
    parts = []
    buf = ""
    for ch in cleaned:
        buf += ch
        if ch in ".!?":
            parts.append(buf.strip())
            buf = ""
        if len(parts) >= 18:
            break
    if buf.strip() and len(parts) < 18:
        parts.append(buf.strip())

    # Take leading + a few long-ish sentences
    ranked = sorted(parts, key=len, reverse=True)
    keep = []
    if parts:
        keep.append(parts[0])
    for s in ranked:
        if s not in keep:
            keep.append(s)
        if len(keep) >= max_bullets:
            break

    out = ["### Summary", *[f"- {s}" for s in keep]]
    return "\n".join(out)

