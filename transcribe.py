#!/usr/bin/env python3
"""
Audio/video transcription script with speaker identification.
Generates an SRT file with speech segments and speaker identity.

Usage:
    python transcribe.py "path/to/video.mp4"
    python transcribe.py "path/to/video.mp4" --output "output.srt"
    python transcribe.py "path/to/video.mp4" --config config.yaml
    python transcribe.py "path/to/video.mp4" --model large-v3
    python transcribe.py "path/to/video.mp4" --no-diarize  # Without speaker identification
"""

# IMPORTANT: Patch torch.load BEFORE any other import for PyTorch 2.6+
# Force weights_only=False for pyannote/omegaconf compatibility
import torch
import torch.serialization
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Always force False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load

import argparse
import sys
from pathlib import Path

from lib import TranscribeConfig, WHISPER_MODELS, transcribe, generate_srt, TranscriptionProgress
from lib.transcribers import TranscriptionError
from lib.progress import step_progress

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file to SRT with speaker identification"
    )
    parser.add_argument("input", help="Audio or video file to transcribe")
    parser.add_argument(
        "-o", "--output",
        help="Output SRT file (default: same name as input)"
    )
    parser.add_argument(
        "-c", "--config",
        help="YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "-m", "--model",
        choices=WHISPER_MODELS,
        help="Whisper model (overrides config)"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker identification"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for pyannote (overrides config)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple mode with faster-whisper (without diarization)"
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TranscribeConfig:
    """Load configuration from YAML then apply CLI overrides."""
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG

    # Load from YAML if file exists
    if config_path.exists():
        config = TranscribeConfig.from_yaml(config_path)
    else:
        config = TranscribeConfig()

    # Apply command line overrides
    if args.model:
        config.model = args.model
    if args.device:
        config.device = args.device
    if args.no_diarize:
        config.diarize = False
    if args.hf_token:
        config.hf_token = args.hf_token

    return config


def main():
    args = parse_args()

    # Verify input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    # Output file
    output_path = args.output or str(input_path.with_suffix(".srt"))

    # Configuration
    config = load_config(args)

    # Create progress manager
    progress = TranscriptionProgress(show_progress=True)

    # Transcription
    try:
        segments = transcribe(str(input_path), config, simple=args.simple, progress=progress)
    except TranscriptionError as e:
        progress.error(str(e))
        sys.exit(1)

    # Generate SRT
    include_speaker = config.diarize and not args.simple
    with step_progress(progress, "generate_srt"):
        generate_srt(segments, output_path, include_speaker=include_speaker)

    # Display final summary
    progress.finish(len(segments), output_path)


if __name__ == "__main__":
    main()
