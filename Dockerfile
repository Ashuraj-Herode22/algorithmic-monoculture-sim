# Lightweight, reproducible scientific container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered terminal logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies using pre-compiled wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, experiments, and tests
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY tests/ ./tests/
COPY pyproject.toml .

# Create output directories for data and figures
RUN mkdir -p data figures

# Default execution: Run mathematical unit tests first, then execute full parameter sweep
CMD ["sh", "-c", "python tests/test_math.py && python experiments/run_experiments.py"]
