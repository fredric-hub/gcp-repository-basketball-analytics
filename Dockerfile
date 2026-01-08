# Use a base image with PyTorch 2.1+ and CUDA 12.1 support
FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest

# ==============================================================================
# FIX: Switch to root user to install system dependencies
# The base image defaults to a non-root user, which causes apt-get to fail.
# ==============================================================================
USER root

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and compiling SAM2
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- SAM2 Installation ---
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

# Expose the port
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "src.main:app"]

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "src.main:app"]
