# Voice-Controlled Local AI Agent (Mem0 Intern Assignment)

Local voice agent with a Streamlit UI: audio/text input → speech-to-text → action planning → safe local tool execution (restricted to `output/`).

## Links (fill these in)
- **Video demo: https://youtu.be/TjJ8u3Jjr0g?si=gqMBe8OiDk2KEz9q
- **Technical article (Substack): https://amukta05.substack.com/p/voice-controlled-local-ai-agent

## Features
- **Audio input**: microphone (browser recorder) or file upload
- **Text input**: bypasses STT for quick testing
- **Intents**: create file, write code, summarize text, general chat
- **Safety**: file operations are restricted to `output/` + optional approval before writes
- **History**: saves past runs to `data/history.jsonl`

## Quickstart

### 1) Create environment & install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Models (local-first)

#### Speech-to-text (faster-whisper)
The app uses `faster-whisper`.

To run fully offline, download the model once (while online), then point the UI to the local folder:

```bash
python scripts/download_whisper_model.py
```

In the Streamlit sidebar:
- **Whisper model (faster-whisper)**: `data/models/Systran__faster-whisper-small`
- **Whisper offline (local files only)**: ON
- **Whisper download root (optional)**: `data/models`

If you prefer environment variables:

```bash
export WHISPER_MODEL=small
export WHISPER_DOWNLOAD_ROOT=data/models
export WHISPER_LOCAL_FILES_ONLY=0
```

#### Planner model (Ollama)
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

## Hardware notes / workarounds

- **Microphone capture**
  - Prefer **Browser (component)** (records in the browser).
  - **Local (PortAudio)** uses `sounddevice` and can fail depending on drivers/permissions. If it fails, use **Upload** or **Text**.

- **Upload disconnects (“Connection lost…”)**
  - Streamlit can disconnect on large uploads. This repo increases Streamlit upload/message limits and defers audio decoding until you click **Run agent**.

## Troubleshooting
- **STT fails with “outgoing traffic has been disabled / local_files_only”**
  - Turn **Whisper offline** OFF (to allow downloads), or set **Whisper model** to `data/models/Systran__faster-whisper-small`.
- **Ollama connection errors**
  - Ensure Ollama is running and `OLLAMA_HOST` is reachable (default: `http://localhost:11434`).

## Repo safety
All file operations are restricted to `output/` by design.

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

