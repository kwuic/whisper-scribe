# Whisper Scribe

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Command-line tool to transcribe audio/video files into SRT subtitles with automatic speaker identification.

## Features

- **Automatic transcription** - Uses OpenAI Whisper via WhisperX
- **Speaker identification** - Automatically detects and labels different speakers (diarization)
- **GPU support** - CUDA acceleration for fast transcriptions
- **Smart fallback** - Automatically switches to lighter models when running out of memory
- **Simple mode** - faster-whisper option for quick transcription without diarization

## Installation

### Prerequisites

- Python 3.9+
- FFmpeg installed on the system
- (Optional) NVIDIA GPU with CUDA for acceleration

### Quick installation

```bash
# Clone the repository
git clone https://github.com/kwuic/whisper-scribe.git
cd whisper-scribe

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure
cp config.yaml.example config.yaml
```

### Installation with pip

```bash
pip install .
```

## Configuration

### HuggingFace Token (required for speaker identification)

1. Create an account on [HuggingFace](https://huggingface.co)
2. Generate a token at [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Accept the pyannote terms of use:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

4. Configure the token:

```bash
# Copy the example file
cp .env.example .env

# Edit and add your token
nano .env  # or your preferred editor
```

Contents of `.env`:
```env
HF_TOKEN=hf_your_token_here
```

Alternative: CLI argument (not recommended, token appears in history)
```bash
python transcribe.py video.mp4 --hf-token "hf_your_token"
```

### config.yaml file

```yaml
model: large-v3      # tiny, base, small, medium, large-v2, large-v3
device: auto         # auto, cuda, cpu
language: fr         # ISO 639-1 language code
diarize: true        # Speaker identification
batch_size: 16       # Reduce if memory error
beam_size: 5
```

### .env file

```env
HF_TOKEN=hf_your_huggingface_token
```

## Usage

### Basic usage

```bash
# Standard transcription with speaker identification
python transcribe.py "video.mp4"

# The video.srt file will be generated automatically
```

### Advanced options

```bash
# Specify output file
python transcribe.py "video.mp4" -o "subtitles.srt"

# Use a lighter model (faster, less accurate)
python transcribe.py "video.mp4" -m medium

# Simple mode without speaker identification (faster)
python transcribe.py "video.mp4" --simple

# Disable speaker identification
python transcribe.py "video.mp4" --no-diarize

# Force CPU usage
python transcribe.py "video.mp4" -d cpu

# Use a specific configuration file
python transcribe.py "video.mp4" -c my_config.yaml
```

### Installed command

If installed via pip:

```bash
video-transcribe "video.mp4" -o "output.srt"
```

## Available models

| Model | Size | Required VRAM | Quality |
|-------|------|---------------|---------|
| tiny | 39M | ~1 GB | Basic |
| base | 74M | ~1 GB | Fair |
| small | 244M | ~2 GB | Good |
| medium | 769M | ~5 GB | Very good |
| large-v2 | 1550M | ~10 GB | Excellent |
| large-v3 | 1550M | ~10 GB | Excellent |

The script automatically falls back to a lighter model when memory is insufficient.

## Output format

Standard SRT file with speaker identification:

```srt
1
00:00:00,000 --> 00:00:02,500
[SPEAKER_00] Hello and welcome.

2
00:00:02,800 --> 00:00:05,200
[SPEAKER_01] Thank you for having me.

3
00:00:05,500 --> 00:00:08,100
[SPEAKER_00] So, let's talk about your project.
```

## Project structure

```
whisper-scribe/
├── transcribe.py              # CLI entry point
├── lib/                       # Internal library
│   ├── __init__.py
│   ├── config.py              # Configuration and types
│   ├── transcribers.py        # Transcription engines
│   ├── progress.py            # Progress display
│   ├── srt.py                 # SRT generation
│   └── utils.py               # Utilities
├── docs/                      # Technical documentation
│   ├── ARCHITECTURE.md        # Project architecture
│   └── API_REFERENCE.md       # Function reference
├── data/                      # Audio/video files (gitignored)
│   └── .gitkeep
├── .env.example               # Secrets template
├── .gitignore
├── .dockerignore
├── config.yaml.example        # Configuration template
├── docker-compose.yml         # Docker orchestration
├── Dockerfile                 # Docker GPU build
├── LICENSE                    # MIT License
├── pyproject.toml             # Python package
├── README.md
└── requirements.txt
```

## Docker (Production)

### Prerequisites

- Docker 20.10+
- NVIDIA Container Toolkit (for GPU)
  ```bash
  # Ubuntu/Debian
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

### Build the image

```bash
docker build -t whisper-scribe .
```

### Usage with Docker Compose (recommended)

```bash
# Configure HuggingFace token
echo "HF_TOKEN=hf_your_token" > .env

# Create the folder for files
mkdir -p data
cp my_video.mp4 data/

# Run transcription
docker compose run --rm transcribe /data/my_video.mp4

# With options
docker compose run --rm transcribe /data/my_video.mp4 -o /data/output.srt -m medium

# CPU only mode
docker compose run --rm transcribe-cpu /data/my_video.mp4
```

### Usage with Docker Run

```bash
# With GPU
docker run --gpus all \
  -e HF_TOKEN="hf_your_token" \
  -v $(pwd)/data:/data \
  whisper-scribe /data/video.mp4 -o /data/output.srt

# Without GPU (CPU)
docker run \
  -e HF_TOKEN="hf_your_token" \
  -v $(pwd)/data:/data \
  whisper-scribe /data/video.mp4 -d cpu
```

### Image size

The image is approximately **8-10 GB** due to CUDA and PyTorch. To reduce:
- Use `--simple` which doesn't require pyannote
- Use CPU mode (lighter image possible with a different base)

## Troubleshooting

### CUDA out of memory error

The script automatically tries lighter models. You can also:

```bash
# Use a smaller model
python transcribe.py video.mp4 -m small

# Or force CPU
python transcribe.py video.mp4 -d cpu
```

### HuggingFace token error

Check that:
1. The token is valid and active
2. You have accepted the terms on the pyannote model pages
3. The token has read permissions

### FFmpeg not found

Install FFmpeg:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (with chocolatey)
choco install ffmpeg
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperX)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio)
