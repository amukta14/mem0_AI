import io
import json
import os
import time
from dataclasses import asdict

# Streamlit tries to write under ~/.streamlit for installation/session metadata.
# In locked-down environments, that can fail; redirect to a workspace-local dir.
_workspace_streamlit_dir = os.path.join(os.path.dirname(__file__), ".streamlit")
os.environ.setdefault("STREAMLIT_CONFIG_DIR", _workspace_streamlit_dir)
# Avoid telemetry initialization that writes an installation id under ~/.streamlit.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.makedirs(_workspace_streamlit_dir, exist_ok=True)

import sounddevice as sd
import numpy as np
import soundfile as sf
import streamlit as st

from src.voice_agent.agent import Agent
from src.voice_agent.audio import record_audio_to_wav_bytes
from src.voice_agent.history import append_history, load_history
from src.voice_agent.types import AgentRunResult


st.set_page_config(page_title="Voice Local AI Agent", layout="wide")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_whisper_env_from_ui() -> None:
    # faster-whisper reads these at model init time (inside stt.py).
    os.environ["WHISPER_LOCAL_FILES_ONLY"] = "1" if st.session_state.get("whisper_local_only") else "0"
    root = (st.session_state.get("whisper_download_root") or "").strip()
    if root:
        os.environ["WHISPER_DOWNLOAD_ROOT"] = root
    else:
        os.environ.pop("WHISPER_DOWNLOAD_ROOT", None)


@st.cache_resource
def _agent() -> Agent:
    return Agent()


def _render_result(result: AgentRunResult) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transcription")
        st.code(result.transcript or "", language=None)
        st.subheader("Detected intent(s)")
        st.json([a.intent for a in result.actions])
    with c2:
        st.subheader("Plan")
        st.json([asdict(a) for a in result.actions])

    st.subheader("Tool execution")
    for idx, step in enumerate(result.steps):
        with st.expander(f"Step {idx+1}: {step.label}", expanded=(idx == 0)):
            st.write(step.status)
            if step.details:
                st.json(step.details)

    st.subheader("Final output")
    st.write(result.final_output)


def _list_input_devices() -> list[tuple[int, str]]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []

    out: list[tuple[int, str]] = []
    for idx, d in enumerate(devices):
        try:
            if int(d.get("max_input_channels") or 0) <= 0:
                continue
            name = str(d.get("name") or f"Device {idx}")
            host = str(d.get("hostapi") or "")
            out.append((idx, f"{idx}: {name}" + (f" (hostapi {host})" if host else "")))
        except Exception:
            continue
    return out


def _wav_bytes_from_float32(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


st.title("Voice-Controlled Local AI Agent")
st.caption("Mic/file audio → STT → LLM intent planner → safe local tools (restricted to output/) → full pipeline display")

agent = _agent()

with st.sidebar:
    st.subheader("Settings")
    require_confirmation = _bool_env("REQUIRE_CONFIRMATION", True)
    require_confirmation = st.toggle("Confirm before file operations", value=require_confirmation)
    st.session_state["require_confirmation"] = require_confirmation

    st.divider()
    st.subheader("Models")
    st.text_input("Whisper model (faster-whisper)", value=os.getenv("WHISPER_MODEL", "small"), key="whisper_model")
    whisper_local_only = _bool_env("WHISPER_LOCAL_FILES_ONLY", False)
    whisper_local_only = st.toggle("Whisper offline (local files only)", value=whisper_local_only)
    st.session_state["whisper_local_only"] = whisper_local_only
    st.text_input(
        "Whisper download root (optional)",
        value=os.getenv("WHISPER_DOWNLOAD_ROOT", "data/models"),
        key="whisper_download_root",
        help="If set, faster-whisper will store/load models from this directory.",
    )
    st.text_input("Ollama host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"), key="ollama_host")
    st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), key="ollama_model")
    st.caption("Changes apply on next run.")

tab1, tab2, tab3 = st.tabs(["Run", "History", "About"])

with tab1:
    st.subheader("Audio input")
    mode = st.radio("Choose input method", ["Microphone", "Upload", "Text"], horizontal=True)

    wav_bytes: bytes | None = None
    wav_sr: int | None = None
    typed_text: str | None = None
    uploaded_file = None

    _apply_whisper_env_from_ui()

    if mode == "Microphone":
        st.caption("Microphone capture runs in your browser (recommended) or via local PortAudio fallback.")
        c1, c2 = st.columns([1, 2])
        with c1:
            seconds = st.slider("Record seconds", 1, 20, 6)
            sr = st.selectbox("Sample rate", [16000, 22050, 44100], index=0)

            mic_backend = st.radio(
                "Microphone backend",
                ["Browser (component)", "Local (PortAudio)"],
                horizontal=True,
                index=0,
            )

            if mic_backend == "Browser (component)":
                st.info("Record below, then click Run agent.")
                try:
                    from st_audiorec import st_audiorec  # type: ignore

                    wav_audio_data = st_audiorec()
                    if wav_audio_data:
                        # streamlit-audiorec returns WAV bytes (16-bit, 44.1kHz)
                        wav_bytes = bytes(wav_audio_data)
                        wav_sr = 44100
                        st.audio(wav_bytes, format="audio/wav")
                except Exception as e:
                    st.error(
                        "Audio recorder component failed to load. "
                        "If you're using a strict browser/privacy setup, try a different browser or use Upload.\n\n"
                        f"Details: {e}"
                    )
            else:
                input_devices = _list_input_devices()
                device_labels = ["Auto"] + [label for _, label in input_devices]
                device_choice = st.selectbox("Input device (PortAudio)", device_labels, index=0)
                chosen_device: int | None = None
                if device_choice != "Auto":
                    chosen_device = int(device_choice.split(":", 1)[0])
                if st.button("Record (PortAudio)"):
                    with st.spinner("Recording..."):
                        try:
                            wav_bytes = record_audio_to_wav_bytes(
                                seconds=seconds, sample_rate=sr, device=chosen_device
                            )
                            wav_sr = sr
                        except Exception as e:
                            wav_bytes = None
                            wav_sr = None
                            st.error(
                                "Microphone recording failed.\n\n"
                                f"Details: {e}"
                            )
        with c2:
            st.caption("Tip: speak clearly. Example: “Create a Python file retry.py with a retry function.”")
    else:
        if mode == "Upload":
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=["wav", "mp3", "m4a", "flac", "ogg"],
                help="Tip: decoding happens when you click Run agent (keeps uploads stable).",
            )
            if uploaded_file is not None:
                st.caption(f"Selected: `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")
        else:
            typed_text = st.text_area(
                "Type your command (bypasses microphone + STT)",
                placeholder="Example: Create a Python file output/retry.py with a retry function.",
                height=120,
            )

    st.divider()
    st.subheader("Execute")

    # Persist the last run across reruns so approval buttons work.
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    run_clicked = st.button("Run agent", type="primary", key="run_agent")
    if run_clicked and mode == "Text":
        started = time.time()
        with st.spinner("Running pipeline (plan → tools)…"):
            result = agent.run_text(
                transcript=typed_text or "",
                ollama_host=st.session_state["ollama_host"],
                ollama_model=st.session_state["ollama_model"],
                require_confirmation=st.session_state["require_confirmation"],
            )
        elapsed = time.time() - started
        append_history(result)
        st.session_state["last_result"] = result
        st.success(f"Done in {elapsed:.2f}s")
        _render_result(result)

    if run_clicked and mode != "Text" and wav_bytes is not None:
        started = time.time()
        with st.spinner("Running pipeline (STT → plan → tools)…"):
            result = agent.run(
                wav_bytes=wav_bytes,
                stt_sample_rate=wav_sr,
                whisper_model=st.session_state["whisper_model"],
                ollama_host=st.session_state["ollama_host"],
                ollama_model=st.session_state["ollama_model"],
                require_confirmation=st.session_state["require_confirmation"],
            )
        elapsed = time.time() - started
        append_history(result)
        st.session_state["last_result"] = result
        st.success(f"Done in {elapsed:.2f}s")
        _render_result(result)

    if run_clicked and mode == "Upload" and wav_bytes is None:
        if uploaded_file is None:
            st.warning("Please choose an audio file to upload first.")
        else:
            started = time.time()
            with st.spinner("Preparing upload (decode → WAV)…"):
                raw = uploaded_file.getvalue()
                # Normalize to WAV bytes if possible (faster-whisper accepts file-like audio; WAV is safest)
                try:
                    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                    buf = io.BytesIO()
                    sf.write(buf, data, sr, format="WAV")
                    wav_bytes = buf.getvalue()
                    wav_sr = sr
                except Exception:
                    wav_bytes = raw
                    wav_sr = None

            with st.spinner("Running pipeline (STT → plan → tools)…"):
                result = agent.run(
                    wav_bytes=wav_bytes,
                    stt_sample_rate=wav_sr,
                    whisper_model=st.session_state["whisper_model"],
                    ollama_host=st.session_state["ollama_host"],
                    ollama_model=st.session_state["ollama_model"],
                    require_confirmation=st.session_state["require_confirmation"],
                )
            elapsed = time.time() - started
            append_history(result)
            st.session_state["last_result"] = result
            st.success(f"Done in {elapsed:.2f}s")
            _render_result(result)

    last_result = st.session_state.get("last_result")
    if last_result is not None and getattr(last_result, "needs_confirmation", False):
        st.warning("This plan includes file operations. Approve to execute.")
        if st.button("Approve and execute now", key="approve_execute"):
            with st.spinner("Executing approved file operations…"):
                executed = agent.execute_approved(last_result)
            append_history(executed)
            st.session_state["last_result"] = executed
            _render_result(executed)

with tab2:
    st.subheader("Session memory (persistent)")
    rows = load_history()
    if not rows:
        st.info("No history yet. Run the agent to see actions and outputs recorded here.")
    else:
        st.caption(f"{len(rows)} runs recorded in data/history.jsonl")
        for i, r in enumerate(reversed(rows[-20:])):
            with st.expander(f"{len(rows)-i}. {r.get('transcript','(no transcript)')[:80]}", expanded=False):
                st.json(r)

with tab3:
    st.subheader("How it works")
    st.markdown(
        """
**Pipeline**
- Audio input (mic or upload)
- Local STT (faster-whisper) → transcript
- LLM planner (Ollama) → JSON action plan
- Safe tools: create/write files under `output/`, summarize, chat
- UI shows transcript, intents, plan, execution steps, and final output

**Safety**
- Hard restriction to `output/` for any file operations
- Optional approval gate before writes
        """
    )

