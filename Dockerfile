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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy built frontend from stage 1
COPY --from=frontend-build /frontend/dist /app/utils/app_utils/frontend/dist

# Create temp directory
RUN mkdir -p /app/temp_processing

# Expose port (HF Spaces expects 7860 by default, but we use 8000)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "app.py"]
