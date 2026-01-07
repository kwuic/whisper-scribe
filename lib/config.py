"""Configuration and types for transcription."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "fr"


@dataclass
class Segment:
    """A transcription segment."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class TranscribeConfig:
    """Configuration for transcription."""
    model: str = DEFAULT_MODEL
    device: str = "auto"
    language: str = DEFAULT_LANGUAGE
    diarize: bool = True
    hf_token: Optional[str] = None
    batch_size: int = 16
    beam_size: int = 5

    def get_device(self) -> str:
        """Return the device to use (cuda or cpu)."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def get_compute_type(self) -> str:
        """Return the optimal compute type for the device."""
        return "float16" if self.get_device() == "cuda" else "int8"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TranscribeConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Convert empty hf_token to None
        if data.get("hf_token") == "":
            data["hf_token"] = None

        return cls(
            model=data.get("model", DEFAULT_MODEL),
            device=data.get("device", "auto"),
            language=data.get("language", DEFAULT_LANGUAGE),
            diarize=data.get("diarize", True),
            hf_token=data.get("hf_token"),
            batch_size=data.get("batch_size", 16),
            beam_size=data.get("beam_size", 5),
        )
