from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional


def transcribe_wav_bytes(wav_bytes: bytes, model_name: str = "small", sample_rate: Optional[int] = None) -> str:
    """
    Local STT via faster-whisper.
    Input must be audio bytes; for non-wav uploads the UI attempts to normalize to WAV.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not available. Install requirements and try again."
        ) from e

    if not wav_bytes:
        raise ValueError("No audio provided")

    # faster-whisper can take file-like objects
    audio_fp = io.BytesIO(wav_bytes)
    model_name = (model_name or "").strip()
    # Common macOS typo: missing leading "/" in absolute paths shown in UI.
    if model_name.startswith("Users/"):
        candidate = "/" + model_name
        if Path(candidate).exists():
            model_name = candidate

    local_only = os.getenv("WHISPER_LOCAL_FILES_ONLY", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT") or None

    try:
        model = WhisperModel(
            model_name,
            device="auto",
            compute_type="auto",
            download_root=download_root,
            local_files_only=local_only,
        )
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            raise RuntimeError(
                "Speech-to-text failed because the Whisper model could not be downloaded (403 Forbidden).\n\n"
                "Fix options:\n"
                "- Set WHISPER_MODEL to a local model path (downloaded earlier), and set WHISPER_LOCAL_FILES_ONLY=1\n"
                "- Or run in an environment with Hugging Face access.\n"
                "- Or use Text mode / Upload mode.\n\n"
                f"Original error: {e}"
            ) from e
        raise
    segments, info = model.transcribe(audio_fp, vad_filter=True)

    parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
    transcript = " ".join(parts).strip()

    # Some short inputs may produce empty transcript; surface as a helpful error.
    if not transcript:
        lang = getattr(info, "language", None)
        raise RuntimeError(f"Could not transcribe audio (detected_language={lang}). Try clearer audio.")

    return transcript

