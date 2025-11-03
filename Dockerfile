# Multi-stage Dockerfile for Semantic Kernel UI
# Security-hardened, optimized for production

# ============================================================================
# Stage 1: Base Python image with security updates
# ============================================================================
FROM python:3.11-slim AS base

# Install security updates and essential tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /app/memory /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app

# ============================================================================
# Stage 2: Dependencies builder (with optional Tesseract OCR)
# ============================================================================
FROM base AS builder

# Build argument to control Tesseract installation
ARG INSTALL_TESSERACT=true

# Install system dependencies for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Tesseract OCR if enabled (optional for OCR functionality)
RUN if [ "$INSTALL_TESSERACT" = "true" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            tesseract-ocr \
            tesseract-ocr-eng \
            tesseract-ocr-fin \
            tesseract-ocr-swe \
            libtesseract-dev && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Install OCR dependencies if Tesseract is enabled
RUN if [ "$INSTALL_TESSERACT" = "true" ]; then \
        pip install --no-cache-dir pytesseract pillow pdf2image; \
    fi

# ============================================================================
# Stage 3: Production runtime (minimal, security-hardened)
# ============================================================================
FROM base AS production

# Build argument for Tesseract
ARG INSTALL_TESSERACT=true

# Copy Tesseract from builder if enabled
RUN if [ "$INSTALL_TESSERACT" = "true" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            tesseract-ocr \
            tesseract-ocr-eng \
            tesseract-ocr-fin \
            tesseract-ocr-swe && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser run.py /app/
COPY --chown=appuser:appuser .env.example /app/.env.example

# Set proper permissions
RUN chmod -R 755 /app/src && \
    chmod 644 /app/run.py && \
    chmod 700 /app/memory && \
    chmod 700 /app/logs

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_TITLE="Semantic Kernel UI" \
    MEMORY_PERSIST_DIRECTORY=/app/memory \
    DEBUG_MODE=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "src/semantic_kernel_ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]

# ============================================================================
# Stage 4: Development image (with dev tools)
# ============================================================================
FROM production AS development

USER root

# Install development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        less \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipython

# Copy tests
COPY --chown=appuser:appuser tests/ /app/tests/
COPY --chown=appuser:appuser pytest.ini /app/

USER appuser

# Override CMD for development (allows running tests)
CMD ["streamlit", "run", "src/semantic_kernel_ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.runOnSave=true", \
     "--server.fileWatcherType=auto"]
