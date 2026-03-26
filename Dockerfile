FROM nvcr.io/nvidia/cuda-quantum:0.9.1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src ./src/
COPY evaluation ./evaluation/
COPY tests ./tests/
COPY examples ./examples/
COPY README.md ./

# Install dependencies using system Python (from cuda-quantum image)
RUN uv sync --python $(which python3)

ENV PATH="/app/.venv/bin:$PATH"
