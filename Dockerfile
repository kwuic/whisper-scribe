# =============================================================================
# Whisper Scribe - Docker with CUDA GPU support
# =============================================================================
# Build:  docker build -t whisper-scribe .
# Run:    docker run --gpus all -v ./data:/data whisper-scribe /data/video.mp4
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base with CUDA
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Copy dependency files
COPY requirements.txt .

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    whisperx>=3.1.0 \
    faster-whisper>=1.0.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    rich>=13.0.0

# -----------------------------------------------------------------------------
# Stage 3: Runtime - Optimized final image
# -----------------------------------------------------------------------------
FROM base AS runtime

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Working directory
WORKDIR /app

# Copy source code
COPY --chown=appuser:appuser lib/ ./lib/
COPY --chown=appuser:appuser transcribe.py .
COPY --chown=appuser:appuser config.yaml.example ./config.yaml

# Directory for input/output files
RUN mkdir -p /data && chown appuser:appuser /data
VOLUME ["/data"]

# Switch to non-root user
USER appuser

# Default environment variables
ENV HF_TOKEN=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Entry point
ENTRYPOINT ["python", "transcribe.py"]

# Default help
CMD ["--help"]

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
LABEL org.opencontainers.image.title="Whisper Scribe"
LABEL org.opencontainers.image.description="Audio/video transcription with speaker identification"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"
