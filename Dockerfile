# Build stage
FROM python:3.11-slim AS builder

# Install uv via pip
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files and README (needed for hatchling build)
COPY pyproject.toml uv.lock README.md ./

# Copy source code needed for editable install
COPY eval/ ./eval/
COPY webapp/ ./webapp/

# Install dependencies
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

# Install uv in runtime image
RUN pip install uv

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Expose port
EXPOSE 5001

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "python", "webapp/app.py"]
