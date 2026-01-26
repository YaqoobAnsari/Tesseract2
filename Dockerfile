# === Stage 1: Build React frontend with Vite ===
FROM node:18-alpine AS frontend-build
WORKDIR /frontend
COPY utils/app_utils/frontend/package.json utils/app_utils/frontend/package-lock.json* ./
RUN npm install
COPY utils/app_utils/frontend/ ./
RUN npm run build

# === Stage 2: Python runtime ===
FROM python:3.11-slim

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~2GB vs CUDA wheels)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from stage 1
COPY --from=frontend-build /frontend/dist /app/utils/app_utils/frontend/dist

# Create writable directories and set permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/temp_processing /app/Results && \
    chmod -R 777 /app/temp_processing /app/Results

# Switch to non-root user (HF Spaces requirement)
USER 1000

# Environment
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "app.py"]
