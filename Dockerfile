# Single-stage Dockerfile using slim Python base
FROM python:3.11-slim

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    NEXT_TELEMETRY_DISABLED=1

ARG APP_ENV=production
ENV NODE_ENV=$APP_ENV

WORKDIR /app

# Install minimal tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates wget curl gnupg lsb-release software-properties-common graphviz && \
    rm -rf /var/lib/apt/lists/*

# Add LLVM 22 repository
RUN curl -L https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get -y install llvm-22-dev llvm-22-tools mlir-22-tools && \
    rm -rf /var/lib/apt/lists/*

# Add Node.js 20 repository and install runtime deps
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=10001:10001 . /app

# Install JS dependencies, then install 'concurrently' globally
WORKDIR /app
RUN npm install && \
    npm install -g concurrently

# Create Python venv and install Python packages
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --pre torch-mlir torchvision \
      --extra-index-url=https://download.pytorch.org/whl/nightly/cpu \
      -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels && \
    /opt/venv/bin/pip install triton fastapi uvicorn pytest httpx PyPDF2

# Create non-root user and fix permissions
RUN useradd -u 10001 -m --shell /usr/sbin/nologin appuser && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.cache /app
USER appuser

# Update PATH for venv and LLVM
ENV PATH="/opt/venv/bin:/usr/lib/llvm-22/bin:$PATH"

# Expose ports and add healthcheck
EXPOSE 3000 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# Default to interactive shell
CMD ["npm", "run", "start:all"]
