# RunPod Serverless Worker for Wan 2.2 Animate
# Using RunPod PyTorch 2.4.0 base - required for torch.nn.RMSNorm (diffusers 0.36+)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

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
