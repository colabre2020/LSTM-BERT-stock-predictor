FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    # Install PyTorch with MPS support
    pip install --no-cache-dir torch torchvision torchaudio

# Copy the application
COPY . .

# Create necessary directories
RUN mkdir -p models/saved data

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_DEVICE=mps
ENV FLASK_APP=src.dashboard.main
ENV FLASK_ENV=development
ENV WEAVIATE_URL=http://weaviate:8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Initialize the application and start the server
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]