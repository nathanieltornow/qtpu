FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python and basics
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src/
COPY vendor ./vendor/
COPY evaluation ./evaluation/
COPY logs ./logs/
COPY examples ./examples/
COPY README.md ./

# Install the package with all dependencies + GPU libraries
# cupy-cuda12x: GPU array library
# cuquantum-python-cu12: NVIDIA cuQuantum for GPU-accelerated tensor network contractions
RUN uv sync && uv pip install cupy-cuda12x cuquantum-python-cu12

# Use the venv python
ENV PATH="/app/.venv/bin:$PATH"
