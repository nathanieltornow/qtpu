FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install Python and basics
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv git curl gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src/
COPY evaluation ./evaluation/
COPY tests ./tests/
COPY examples ./examples/
COPY README.md ./

# Install dependencies + cuda-quantum
RUN uv sync && uv pip install cuda-quantum-cu12

ENV PATH="/app/.venv/bin:$PATH"
