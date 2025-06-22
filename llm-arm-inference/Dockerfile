# Multi-stage Dockerfile for ARM-optimized LLM inference
ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM=linux/amd64

FROM --platform=$BUILDPLATFORM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r llm && useradd -r -g llm llm

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY requirements.txt .

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache && \
    chown -R llm:llm /app

# Architecture-specific optimizations
ARG TARGETPLATFORM
RUN echo "Building for platform: $TARGETPLATFORM"

# Install architecture-specific llama-cpp-python
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        echo "Installing llama-cpp-python for ARM64 with NEON support..."; \
        pip install --no-cache-dir llama-cpp-python==0.2.20 \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; \
    else \
        echo "Installing llama-cpp-python for AMD64..."; \
        pip install --no-cache-dir llama-cpp-python==0.2.20; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_DIR=/app/cache
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers

# ARM-specific optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER llm

# Default command
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
