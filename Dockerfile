FROM python:3.11-slim

# Build arguments for version tracking
ARG BUILD_COMMIT_ARG=unknown
ARG BUILD_BRANCH_ARG=unknown
ARG BUILD_TIMESTAMP_ARG=unknown

# Set environment variables for build info
ENV BUILD_COMMIT=$BUILD_COMMIT_ARG
ENV BUILD_BRANCH=$BUILD_BRANCH_ARG
ENV BUILD_TIMESTAMP=$BUILD_TIMESTAMP_ARG

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ templates/

# Create data directory
RUN mkdir -p /app/data/boards /app/data/posts /app/data/tracking

# Expose port
EXPOSE 5000

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)"

# Run the application
CMD ["python", "app.py"]
