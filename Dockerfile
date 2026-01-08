# ==============================================================================
# 1. Base Image Update
#    We switch to PyTorch 2.4 to get Python 3.10+ (Required by SAM2)
# ==============================================================================
FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-4:latest

# ==============================================================================
# 2. Permission Fix
#    The Vertex AI image defaults to a non-root user 'monitor'.
#    We must switch to root to install system dependencies via apt-get.
# ==============================================================================
USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg is required for supervision/opencv)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# Note: We do NOT re-install torch here to avoid overwriting the optimized 
# version included in the base image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- SAM2 Installation ---
# We clone and install the specific real-time fork used in the notebook
RUN git clone https://github.com/Gy920/segment-anything-2-real-time.git sam2_repo \
    && cd sam2_repo \
    && pip install -e . \
    && python setup.py build_ext --inplace

# Download SAM2 Checkpoints
RUN cd sam2_repo/checkpoints && ./download_ckpts.sh
ENV SAM2_CHECKPOINT_PATH=/app/sam2_repo/checkpoints/sam2.1_hiera_large.pt
ENV SAM2_CONFIG_PATH=/app/sam2_repo/configs/sam2.1/sam2.1_hiera_l.yaml

# Copy source code
COPY src/ ./src/

EXPOSE 8080

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "src.main:app"]