"""Progress module for real-time progress display."""

import time
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table


console = Console()

# Transcription steps with their relative weights
STEPS = {
    "load_model": ("Loading model", 10),
    "load_audio": ("Loading audio", 5),
    "transcribe": ("Transcription", 40),
    "align": ("Timestamp alignment", 15),
    "diarize": ("Speaker identification", 25),
    "generate_srt": ("Generating SRT", 5),
}


class TranscriptionProgress:
    """Progress manager for transcription."""

    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self.start_time = None
        self.current_step = None
        self._progress = None
        self._task_id = None

    def start(self, filename: str, config_info: dict):
        """Start progress tracking."""
        self.start_time = time.time()

        if not self.show_progress:
            return

        # Display configuration information
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("File", filename)
        table.add_row("Model", config_info.get("model", "?"))
        table.add_row("Device", config_info.get("device", "?"))
        table.add_row("Language", config_info.get("language", "?"))
        table.add_row("Diarization", "Yes" if config_info.get("diarize") else "No")

        console.print(Panel(table, title="[bold blue]Configuration", border_style="blue"))
        console.print()

    def step(self, step_name: str, message: Optional[str] = None):
        """Start a new step."""
        self.current_step = step_name

        if not self.show_progress:
            return

        step_info = STEPS.get(step_name, (step_name, 0))
        display_name = step_info[0]
        msg = message or display_name

        console.print(f"[bold cyan]→[/] [yellow]{msg}[/]...")

    def step_done(self, step_name: str, message: Optional[str] = None):
        """Mark a step as completed."""
        if not self.show_progress:
            return

        step_info = STEPS.get(step_name, (step_name, 0))
        display_name = step_info[0]
        msg = message or display_name
        elapsed = self._format_elapsed()

        console.print(f"[bold green]✓[/] {msg} [dim]({elapsed})[/]")

    def info(self, message: str):
        """Display an informational message."""
        if self.show_progress:
            console.print(f"  [dim]{message}[/]")

    def warning(self, message: str):
        """Display a warning."""
        if self.show_progress:
            console.print(f"  [bold yellow]⚠[/] {message}")

    def error(self, message: str):
        """Display an error."""
        console.print(f"[bold red]✗[/] {message}")

    def finish(self, segment_count: int, output_path: str):
        """Finish tracking and display summary."""
        if not self.show_progress:
            return

        total_time = time.time() - self.start_time if self.start_time else 0

        console.print()
        console.print(Panel(
            f"[bold green]Done![/]\n\n"
            f"[cyan]Segments:[/] {segment_count}\n"
            f"[cyan]Duration:[/] {self._format_time(total_time)}\n"
            f"[cyan]File:[/] {output_path}",
            title="[bold green]Result",
            border_style="green"
        ))

    def _format_elapsed(self) -> str:
        """Format elapsed time since start."""
        if not self.start_time:
            return "0s"
        elapsed = time.time() - self.start_time
        return self._format_time(elapsed)

    def _format_time(self, seconds: float) -> str:
        """Format duration in readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.0f}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"


@contextmanager
def step_progress(progress: TranscriptionProgress, step_name: str, message: Optional[str] = None):
    """Context manager to track a step."""
    progress.step(step_name, message)
    try:
        yield
    finally:
        progress.step_done(step_name, message)


# Global instance for easy access
_current_progress: Optional[TranscriptionProgress] = None


def get_progress() -> Optional[TranscriptionProgress]:
    """Return the current progress instance."""
    return _current_progress


def set_progress(progress: Optional[TranscriptionProgress]):
    """Set the current progress instance."""
    global _current_progress
    _current_progress = progress
