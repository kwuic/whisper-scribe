"""Utilities for transcription."""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    # Look for .env in the project folder
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system variables


def patch_torch_load():
    """
    Patch torch.load for PyTorch 2.6+ with pyannote/omegaconf.
    Note: The main patch is in transcribe.py, this function is kept
    for compatibility if the module is used directly.
    """
    pass  # Patch applied in transcribe.py


def get_hf_token(provided_token: Optional[str] = None) -> Optional[str]:
    """Get the HuggingFace token from argument or environment."""
    return provided_token or os.environ.get("HF_TOKEN")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
