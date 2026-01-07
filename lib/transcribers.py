"""Audio/video transcription functions."""

import sys
from typing import List, Optional

from .config import TranscribeConfig, Segment
from .utils import get_hf_token
from .progress import TranscriptionProgress, step_progress, set_progress, get_progress


class TranscriptionError(Exception):
    """Error during transcription."""
    pass


# Fallback order for models in case of memory error
MODEL_FALLBACK_ORDER = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]


def transcribe(
    audio_path: str,
    config: TranscribeConfig,
    simple: bool = False,
    progress: Optional[TranscriptionProgress] = None
) -> List[Segment]:
    """
    Main entry point for transcription.

    Args:
        audio_path: Path to the audio/video file
        config: Transcription configuration
        simple: Use faster-whisper (without diarization)
        progress: Progress instance (optional)

    Returns:
        List of transcribed segments
    """
    # Use provided progress or create one
    if progress is None:
        progress = TranscriptionProgress(show_progress=True)
    set_progress(progress)

    # Display configuration
    progress.start(audio_path, {
        "model": config.model,
        "device": config.get_device(),
        "language": config.language,
        "diarize": config.diarize and not simple,
    })

    if simple:
        return _transcribe_faster_whisper(audio_path, config, progress)
    return _transcribe_whisperx_with_fallback(audio_path, config, progress)


def _transcribe_whisperx_with_fallback(
    audio_path: str,
    config: TranscribeConfig,
    progress: TranscriptionProgress
) -> List[Segment]:
    """Transcription with automatic fallback on memory error."""
    current_model = config.model
    tried_models = []

    while True:
        try:
            # Update model in config
            config.model = current_model
            return _transcribe_whisperx(audio_path, config, progress)

        except (RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            is_oom = any(x in error_msg for x in [
                "out of memory",
                "cuda out of memory",
                "oom",
                "not enough memory",
                "allocat",
            ])

            if not is_oom:
                raise

            tried_models.append(current_model)
            progress.warning(f"Insufficient memory with model '{current_model}'")

            # Find next model to try
            next_model = _get_next_fallback_model(current_model, tried_models)

            if next_model is None:
                progress.error("All models failed due to insufficient memory")
                raise TranscriptionError(
                    f"Unable to transcribe: insufficient memory with all tried models: {tried_models}"
                )

            progress.warning(f"Falling back to model '{next_model}'")
            current_model = next_model

            # Free CUDA memory before retrying
            _clear_cuda_memory()


def _get_next_fallback_model(current: str, tried: List[str]) -> Optional[str]:
    """Find the next lighter model to try."""
    try:
        current_idx = MODEL_FALLBACK_ORDER.index(current)
    except ValueError:
        current_idx = -1

    for model in MODEL_FALLBACK_ORDER[current_idx + 1:]:
        if model not in tried:
            return model
    return None


def _clear_cuda_memory():
    """Free CUDA memory."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _transcribe_whisperx(
    audio_path: str,
    config: TranscribeConfig,
    progress: TranscriptionProgress
) -> List[Segment]:
    """Transcription with WhisperX and diarization."""
    try:
        import whisperx
    except ImportError:
        raise TranscriptionError(
            "whisperx not installed. Install it with: pip install whisperx"
        )

    device = config.get_device()
    compute_type = config.get_compute_type()

    # Load model
    with step_progress(progress, "load_model", f"Loading model {config.model}"):
        model = whisperx.load_model(config.model, device, compute_type=compute_type)

    # Load audio
    with step_progress(progress, "load_audio"):
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / 16000  # 16kHz sample rate
        progress.info(f"Duration: {duration / 60:.1f} minutes")

    # Transcription
    with step_progress(progress, "transcribe"):
        result = model.transcribe(audio, batch_size=config.batch_size, language=config.language)
        progress.info(f"Detected language: {result.get('language', config.language)}")
        progress.info(f"Raw segments: {len(result.get('segments', []))}")

    # Free model to save memory
    del model
    _clear_cuda_memory()

    # Alignment for precise timestamps
    with step_progress(progress, "align"):
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False
        )
        del model_a
        _clear_cuda_memory()

    # Diarization
    if config.diarize:
        result = _apply_diarization(audio, result, config, device, progress)

    return _convert_to_segments(result["segments"])


def _apply_diarization(
    audio,
    result: dict,
    config: TranscribeConfig,
    device: str,
    progress: TranscriptionProgress
) -> dict:
    """Apply diarization to transcription result."""
    import whisperx

    hf_token = get_hf_token(config.hf_token)

    if not hf_token:
        progress.warning("HuggingFace token not provided")
        progress.info("Diarization requires a HuggingFace token.")
        progress.info("1. Create an account at https://huggingface.co")
        progress.info("2. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        progress.info("3. Set HF_TOKEN or use --hf-token")
        progress.warning("Continuing without speaker identification")
        return result

    with step_progress(progress, "diarize"):
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Count unique speakers
            speakers = set()
            for seg in result.get("segments", []):
                if seg.get("speaker"):
                    speakers.add(seg["speaker"])
            if speakers:
                progress.info(f"Speakers identified: {len(speakers)}")

            del diarize_model
            _clear_cuda_memory()

        except Exception as e:
            progress.warning(f"Diarization error: {e}")
            progress.warning("Continuing without speaker identification")

    return result


def _transcribe_faster_whisper(
    audio_path: str,
    config: TranscribeConfig,
    progress: TranscriptionProgress
) -> List[Segment]:
    """Simple transcription with faster-whisper (without diarization)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(
            "faster-whisper not installed. Install it with: pip install faster-whisper"
        )

    device = config.get_device()
    compute_type = config.get_compute_type()

    # Load model
    with step_progress(progress, "load_model", f"Loading model {config.model}"):
        model = WhisperModel(config.model, device=device, compute_type=compute_type)

    # Transcription
    with step_progress(progress, "transcribe"):
        segments_iter, info = model.transcribe(
            audio_path, language=config.language, beam_size=config.beam_size
        )
        progress.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")

        segments = [
            Segment(start=seg.start, end=seg.end, text=seg.text)
            for seg in segments_iter
        ]
        progress.info(f"Segments: {len(segments)}")

    return segments


def _convert_to_segments(raw_segments: list) -> List[Segment]:
    """Convert raw segments to Segment objects."""
    return [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            speaker=seg.get("speaker"),
        )
        for seg in raw_segments
    ]
