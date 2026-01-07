"""
Module de transcription audio/vidéo avec identification des locuteurs.
"""

from .config import TranscribeConfig, Segment, WHISPER_MODELS
from .transcribers import transcribe
from .srt import generate_srt
from .progress import TranscriptionProgress

__all__ = [
    "TranscribeConfig",
    "Segment",
    "WHISPER_MODELS",
    "transcribe",
    "generate_srt",
    "TranscriptionProgress",
]
