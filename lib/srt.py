"""SRT file generation."""

from typing import List

from .config import Segment
from .utils import format_timestamp


def generate_srt(
    segments: List[Segment],
    output_path: str,
    include_speaker: bool = True
) -> None:
    """
    Generate an SRT file from transcribed segments.

    Args:
        segments: List of segments to write
        output_path: Path to the output SRT file
        include_speaker: Include speaker identity in the text
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()

            if include_speaker and segment.speaker:
                text = f"[{segment.speaker}] {text}"

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")

    print(f"SRT file generated: {output_path}")
