# Audio/Video Transcription - Architecture

## Overview

This project is a Python command-line tool for **transcribing audio/video files** to SRT subtitle files. It uses **Whisper** speech recognition technology and offers **speaker identification** (diarization) via **pyannote**.

### Main Features

- Automatic audio/video transcription to SRT
- Speaker identification (diarization)
- GPU (CUDA) and CPU support
- Automatic fallback to lighter models when memory is insufficient
- Configuration via YAML file or CLI arguments
- Simple mode (faster-whisper) without diarization

---

## System Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#818cf8', 'lineColor': '#94a3b8', 'secondaryColor': '#0f172a', 'tertiaryColor': '#1e293b', 'background': '#0f172a', 'mainBkg': '#1e293b', 'nodeBorder': '#6366f1', 'clusterBkg': '#1e293b', 'clusterBorder': '#475569', 'titleColor': '#f8fafc', 'edgeLabelBackground': '#1e293b', 'nodeTextColor': '#f8fafc'}}}%%
flowchart TB
    subgraph CLI["CLI Interface"]
        MAIN[transcribe.py]
        ARGS[Arguments Parser]
        CONFIG_LOADER["Config Loader"]
    end

    subgraph LIB["Library lib/"]
        subgraph Core["Core"]
            CONFIG[config.py<br/>Configuration & Types]
            TRANSCRIBERS[transcribers.py<br/>Transcription Engines]
        end

        subgraph Output["Output"]
            SRT[srt.py<br/>SRT Generator]
        end

        subgraph Support["Support"]
            PROGRESS[progress.py<br/>Progress Display]
            UTILS[utils.py<br/>Utilities]
        end
    end

    subgraph External["External Dependencies"]
        WHISPERX[WhisperX]
        FASTER[faster-whisper]
        PYANNOTE[pyannote<br/>Diarization]
        TORCH[PyTorch/CUDA]
    end

    subgraph IO["Input/Output"]
        YAML[(config.yaml)]
        AUDIO[("Audio/video file")]
        SRTFILE[(".srt file")]
    end

    MAIN --> ARGS
    ARGS --> CONFIG_LOADER
    CONFIG_LOADER --> CONFIG
    YAML --> CONFIG

    MAIN --> TRANSCRIBERS
    TRANSCRIBERS --> WHISPERX
    TRANSCRIBERS --> FASTER
    WHISPERX --> PYANNOTE
    WHISPERX --> TORCH
    FASTER --> TORCH

    TRANSCRIBERS --> PROGRESS
    TRANSCRIBERS --> UTILS

    AUDIO --> TRANSCRIBERS
    TRANSCRIBERS --> SRT
    SRT --> SRTFILE
```

---

## File Structure

```
whisper-scribe/
├── transcribe.py          # Main CLI entry point
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
└── lib/                   # Internal library
    ├── __init__.py        # Public module exports
    ├── config.py          # Configuration and data types
    ├── transcribers.py    # Transcription logic
    ├── progress.py        # Progress display
    ├── srt.py             # SRT file generation
    └── utils.py           # Utility functions
```

---

## Data Flow

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#818cf8', 'lineColor': '#94a3b8', 'secondaryColor': '#0f172a', 'tertiaryColor': '#1e293b', 'actorTextColor': '#f8fafc', 'actorBkg': '#6366f1', 'signalColor': '#94a3b8', 'signalTextColor': '#f8fafc'}}}%%
sequenceDiagram
    participant U as User
    participant CLI as transcribe.py
    participant CFG as Config
    participant TR as Transcribers
    participant WX as WhisperX
    participant PY as Pyannote
    participant SRT as SRT Generator

    U->>CLI: python transcribe.py video.mp4
    CLI->>CFG: Load config.yaml + CLI args
    CFG-->>CLI: TranscribeConfig

    CLI->>TR: transcribe(path, config)

    alt Full mode (WhisperX)
        TR->>WX: load_model()
        WX-->>TR: Whisper Model
        TR->>WX: load_audio()
        WX-->>TR: Normalized audio
        TR->>WX: transcribe()
        WX-->>TR: Raw segments
        TR->>WX: align()
        WX-->>TR: Precise timestamps

        opt Diarization enabled
            TR->>PY: DiarizationPipeline()
            PY-->>TR: Speaker segments
            TR->>WX: assign_word_speakers()
            WX-->>TR: Segments with speakers
        end
    else Simple mode (faster-whisper)
        TR->>TR: WhisperModel.transcribe()
    end

    TR-->>CLI: List[Segment]
    CLI->>SRT: generate_srt(segments)
    SRT-->>CLI: .srt file written
    CLI-->>U: Transcription complete!
```

---

## Data Model

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#818cf8', 'lineColor': '#94a3b8', 'classText': '#f8fafc'}}}%%
classDiagram
    class TranscribeConfig {
        +str model
        +str device
        +str language
        +bool diarize
        +str hf_token
        +int batch_size
        +int beam_size
        +get_device() str
        +get_compute_type() str
        +from_yaml(path) TranscribeConfig
    }

    class Segment {
        +float start
        +float end
        +str text
        +str speaker
        +to_dict() dict
    }

    class TranscriptionProgress {
        +bool show_progress
        +float start_time
        +str current_step
        +start(filename, config_info)
        +step(step_name, message)
        +step_done(step_name, message)
        +info(message)
        +warning(message)
        +error(message)
        +finish(segment_count, output_path)
    }

    class TranscriptionError {
        +str message
    }

    TranscribeConfig --> Segment : produces
    TranscriptionProgress --> TranscribeConfig : uses
```

---

## Transcription Workflow

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#818cf8', 'lineColor': '#94a3b8'}}}%%
stateDiagram-v2
    [*] --> ParseArgs: Start

    ParseArgs --> LoadConfig: Valid arguments
    ParseArgs --> Error: File not found

    LoadConfig --> SelectMode: Config loaded

    SelectMode --> SimpleMode: --simple
    SelectMode --> FullMode: Standard mode

    state SimpleMode {
        [*] --> FW_LoadModel: faster-whisper
        FW_LoadModel --> FW_Transcribe
        FW_Transcribe --> [*]
    }

    state FullMode {
        [*] --> WX_LoadModel: WhisperX
        WX_LoadModel --> WX_LoadAudio
        WX_LoadAudio --> WX_Transcribe
        WX_Transcribe --> WX_Align
        WX_Align --> CheckDiarize

        CheckDiarize --> Diarize: diarize=true
        CheckDiarize --> [*]: diarize=false

        Diarize --> [*]
    }

    SimpleMode --> GenerateSRT
    FullMode --> GenerateSRT

    GenerateSRT --> Finish: SRT written
    Finish --> [*]

    Error --> [*]

    note right of FullMode
        On memory error,
        automatic fallback to
        a lighter model
    end note
```

---

## Memory Fallback Mechanism

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#818cf8', 'lineColor': '#94a3b8'}}}%%
flowchart TD
    START[Start transcription] --> TRY{Try model}

    TRY -->|Success| SUCCESS[Transcription OK]
    TRY -->|Memory error| CHECK{Next model?}

    CHECK -->|Yes| CLEAR[Free CUDA memory]
    CHECK -->|No| FAIL[Failure: all models tried]

    CLEAR --> SWITCH[Switch to lighter model]
    SWITCH --> TRY

    subgraph Models["Fallback order"]
        direction LR
        M1[large-v3] --> M2[large-v2]
        M2 --> M3[medium]
        M3 --> M4[small]
        M4 --> M5[base]
        M5 --> M6[tiny]
    end
```

---

## Component Descriptions

### `transcribe.py` - CLI Entry Point

Main script that orchestrates the entire process:
- Parses command line arguments
- Loads configuration (YAML + CLI overrides)
- Launches transcription
- Handles errors and displays results

**Supported arguments:**
| Argument | Description |
|----------|-------------|
| `input` | Audio/video file to transcribe |
| `-o, --output` | Output SRT file |
| `-c, --config` | YAML configuration file |
| `-m, --model` | Whisper model (tiny, base, small, medium, large-v2, large-v3) |
| `-d, --device` | Device (auto, cuda, cpu) |
| `--no-diarize` | Disable speaker identification |
| `--hf-token` | HuggingFace token for pyannote |
| `--simple` | Simple mode with faster-whisper |

### `lib/config.py` - Configuration

Defines data types and configuration management:
- `TranscribeConfig`: Transcription parameters (model, device, language, etc.)
- `Segment`: A transcription segment (start, end, text, speaker)
- Loading from YAML file

### `lib/transcribers.py` - Transcription Engines

Contains the main transcription logic:
- **Full mode (WhisperX)**: Transcription + alignment + diarization
- **Simple mode (faster-whisper)**: Basic transcription without diarization
- **Automatic fallback**: Switches to lighter models on OOM errors

### `lib/progress.py` - Progress Display

Manages real-time progress display with Rich:
- Progress steps with visual indicators
- Info, warning, error messages
- Final summary with statistics

### `lib/srt.py` - SRT Generation

Generates subtitle files in SRT format:
- Timestamp formatting (HH:MM:SS,mmm)
- Optional speaker name inclusion

### `lib/utils.py` - Utilities

Various helper functions:
- HuggingFace token retrieval
- Timestamp formatting

---

## Configuration (config.yaml)

```yaml
# Whisper model
model: large-v3        # tiny, base, small, medium, large-v2, large-v3

# Compute device
device: auto           # auto, cuda, cpu

# Transcription language
language: fr

# Speaker identification
diarize: true

# HuggingFace token (required for diarization)
hf_token: "hf_xxx..."

# Advanced parameters
batch_size: 16
beam_size: 5
```

---

## External Dependencies

| Package | Role |
|---------|------|
| `torch` | Deep learning framework, CUDA support |
| `whisperx` | Transcription with alignment and diarization |
| `faster-whisper` | Lightweight alternative for simple transcription |
| `pyannote` | Speaker identification (via whisperx) |
| `pyyaml` | Configuration file reading |
| `rich` | Enhanced console display |

---

## Usage Examples

```bash
# Standard transcription with diarization
python transcribe.py "video.mp4"

# Specify output file
python transcribe.py "video.mp4" --output "subtitles.srt"

# Use a lighter model
python transcribe.py "video.mp4" --model medium

# Simple mode (faster, without diarization)
python transcribe.py "video.mp4" --simple

# Disable diarization
python transcribe.py "video.mp4" --no-diarize

# Force CPU usage
python transcribe.py "video.mp4" --device cpu
```
