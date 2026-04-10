from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_for_model(model: str) -> str:
    # faster-whisper uses CTranslate2-converted Whisper weights published by Systran.
    # See: https://huggingface.co/Systran
    if model.startswith(("Systran/", "openai/", "tiny", "base", "small", "medium", "large")):
        # If user passes "small" etc, map to Systran faster-whisper repos.
        if "/" not in model:
            return f"Systran/faster-whisper-{model}"
        return model
    # If it's an explicit local path, we won't use it here.
    return f"Systran/faster-whisper-{model}"


def main() -> int:
    p = argparse.ArgumentParser(description="Download faster-whisper model to local cache directory.")
    p.add_argument("--model", default=os.getenv("WHISPER_MODEL", "small"), help="Model size or HF repo id (default: small)")
    p.add_argument(
        "--download-root",
        default=os.getenv("WHISPER_DOWNLOAD_ROOT", "data/models"),
        help="Directory to store models (default: data/models)",
    )
    p.add_argument(
        "--revision",
        default=os.getenv("WHISPER_REVISION", None),
        help="Optional HF revision/tag/commit",
    )
    args = p.parse_args()

    download_root = Path(args.download_root).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    model = args.model.strip()
    if not model:
        raise SystemExit("--model cannot be empty")

    repo_id = _repo_for_model(model)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is not installed. Run `pip install -r requirements.txt`."
        ) from e

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None

    print(f"Downloading {repo_id} → {download_root}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(download_root / repo_id.replace("/", "__")),
        local_dir_use_symlinks=False,
        token=token,
        revision=args.revision,
    )
    print(f"Done. Local model path:\n{local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

