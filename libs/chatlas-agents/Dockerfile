# ChATLAS Agents Docker Image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY chatlas_agents/ ./chatlas_agents/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Create a non-root user for security
RUN useradd -m -u 1000 chatlas && chown -R chatlas:chatlas /app
USER chatlas

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHATLAS_AGENT_NAME=chatlas-agent

# Expose port if needed (for future API server)
EXPOSE 8000

# Entry point
ENTRYPOINT ["python", "-m", "chatlas_agents.cli"]
CMD ["--help"]
