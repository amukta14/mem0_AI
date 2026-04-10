from __future__ import annotations

import io

import numpy as np
import sounddevice as sd
import soundfile as sf


def _pick_input_device_index() -> int | None:
    """Return a usable input device index, or None if none found.

    PortAudio's default device (-1) can be invalid in headless/permissions
    scenarios, so we proactively select the first device that advertises input
    channels.
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    for idx, d in enumerate(devices):
        try:
            if int(d.get("max_input_channels") or 0) > 0:
                return idx
        except Exception:
            continue
    return None


def record_audio_to_wav_bytes(
    seconds: int, sample_rate: int = 16000, device: int | None = None
) -> bytes:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    frames = int(seconds * sample_rate)
    if device is None:
        device = _pick_input_device_index()
    if device is None:
        raise RuntimeError(
            "No usable microphone input device found. "
            "If you're running headless or without mic permissions, use Upload mode."
        )

    audio = sd.rec(
        frames, samplerate=sample_rate, channels=1, dtype="float32", device=device
    )
    sd.wait()
    audio = np.squeeze(audio, axis=1)

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()

