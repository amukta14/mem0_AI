# Voice-Controlled Local AI Agent (Mem0 Intern Assignment)

This project implements a **voice-controlled AI agent** that accepts audio input (mic or file upload), transcribes it, classifies intent, safely executes local tools restricted to `output/`, and displays the full pipeline in a clean UI.

## Links (fill these in)
- **Video demo (2–3 min, YouTube unlisted)**: `TODO`
- **Technical article (Substack / Dev.to / Medium)**: `TODO`

## What you can do (minimum required intents)
- **Create a file** (in `output/`)
- **Write code** to a new/existing file (in `output/`)
- **Summarize text**
- **General chat**

## “Stand out” additions (still true to the spec)
- **Compound commands**: one utterance can produce multiple actions (e.g., “summarize and save to summary.md”).
- **Human-in-the-loop**: optional approval before any file write happens.
- **Graceful degradation**: clear errors for unintelligible audio / missing models, and fallbacks.
- **Memory**: persistent session history saved locally in `data/history.jsonl`.
- **Transparent pipeline**: UI shows transcription, detected intent(s), planned actions, tool outputs, and final result.

## Quickstart

### 1) Create environment & install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Local models (recommended)

#### Local STT (offline)
This uses `faster-whisper` by default.

If you want a fully-offline setup, download the model once (while online), then point the app to the local folder:

```bash
python scripts/download_whisper_model.py
```

In the Streamlit sidebar set:
- **Whisper model (faster-whisper)**: `data/models/Systran__faster-whisper-small`
- **Whisper offline (local files only)**: ON
- **Whisper download root (optional)**: `data/models`

If you prefer environment variables instead of the UI, you can also set:
```bash
export WHISPER_MODEL=small
export WHISPER_DOWNLOAD_ROOT=data/models
export WHISPER_LOCAL_FILES_ONLY=0
```

You can choose a model size via:
```bash
export WHISPER_MODEL=small
```

#### Local LLM (offline, preferred)
Install Ollama and pull a model:
```bash
ollama pull llama3.1:8b
export OLLAMA_MODEL=llama3.1:8b
```

### 3) Run the UI
```bash
streamlit run app.py
```

## Configuration
- **`WHISPER_MODEL`**: Whisper model name for `faster-whisper` (default: `small`).
- **`OLLAMA_HOST`**: Ollama base URL (default: `http://localhost:11434`).
- **`OLLAMA_MODEL`**: model name (default: `llama3.1:8b`).
- **`REQUIRE_CONFIRMATION`**: `true/false` (default: `true`) to approve file writes.

## Hardware notes + workarounds
This project is designed to run locally on a laptop/desktop. Two practical “gotchas” showed up during development:

- **Microphone capture**
  - **Recommended**: the **Browser (component)** mic backend (records in the browser).
  - **Fallback**: **Local (PortAudio)** uses `sounddevice` + system audio drivers; this can fail on some machines due to missing PortAudio permissions/drivers. If it fails, use **Upload** or **Text** mode.

- **Upload disconnects (“Connection lost…”)**
  - Streamlit can show “Connection lost” when uploads are large or when heavy decoding runs during reruns.
  - This repo includes `.streamlit/config.toml` with higher `maxUploadSize` / `maxMessageSize` and defers audio decoding until you click **Run agent** to keep uploads stable.

## Troubleshooting
- **STT fails with “outgoing traffic has been disabled / local_files_only”**
  - Turn **Whisper offline** OFF (to allow downloads), or set the model path to the local folder:
    - `data/models/Systran__faster-whisper-small`
- **Ollama connection errors**
  - Ensure Ollama is running and `OLLAMA_HOST` is reachable (default: `http://localhost:11434`).

### Optional API fallback
If you cannot run local STT/LLM efficiently, you can add an API fallback by implementing providers in `src/voice_agent/providers.py`.
This repo is wired to prefer **local** execution; the UI will show a clear message if models aren’t available.

## Repo safety
All file operations are **restricted to `output/`** by design (hard guard in the tool layer).

## Deliverables checklist
- **Code**: this repository
- **Video demo (2–3 min)**: show at least two different intents in the UI (e.g., “write code” and “summarize”)
- **Technical article**: describe architecture, model choices, challenges, and (optionally) a mini benchmark

## Architecture (high level)
1. Audio input (mic or file upload)
2. STT → transcription
3. LLM planner → JSON plan (one or multiple actions)
4. Tool execution (safe filesystem tools + summarization/chat)
5. UI renders: transcription, intent(s), actions, outputs, logs, memory

### Key modules
- **UI**: `app.py` (Streamlit)
- **Agent orchestration**: `src/voice_agent/agent.py`
- **STT**: `src/voice_agent/stt.py` (faster-whisper)
- **Planner**: `src/voice_agent/llm_planner.py` (Ollama)
- **Safe filesystem tools**: `src/voice_agent/fs_tools.py` (restricted to `output/`)
- **History/memory**: `src/voice_agent/history.py` → `data/history.jsonl`

