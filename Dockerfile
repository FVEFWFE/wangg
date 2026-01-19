# RunPod Serverless Worker for Wan 2.2 Animate
# Using plain NVIDIA CUDA base image to avoid XPU issues in RunPod PyTorch images
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch from CUDA-only wheel (no XPU support compiled in)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements_runpod.txt .
RUN pip install --no-cache-dir -r requirements_runpod.txt

# Copy handler code
COPY handler.py .

# Set environment variables
ENV MODEL_ID="Wan-AI/Wan2.2-Animate-14B-Diffusers"
ENV HF_HOME="/runpod-volume/huggingface"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface"

# Run the handler
CMD ["python", "-u", "handler.py"]
