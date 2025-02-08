# Use Triton 24.01 as base image
FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Hugging Face libraries and other dependencies
RUN pip3 install \
    transformers==4.46.1 \
    bitsandbytes==0.44.1 \
    tokenizers==0.20.1 \
    accelerate==1.1.0 \
    sentencepiece==0.2.0 \
    protobuf==4.25.5 \
    nvidia-cublas-cu12==12.1.3.1 \
    peft==0.7.1 \
    # Clean pip cache
    && pip3 cache purge

