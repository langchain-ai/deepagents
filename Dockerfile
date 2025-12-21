# ChATLAS Agents Docker Image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install uv

# Copy application code
COPY . /app/

# Install the package globally (not editable)
RUN cd /app/libs/chatlas-agents && \
    uv sync

# Make directories writable for Apptainer compatibility
# Apptainer runs with host user, so we need write permissions
RUN chmod -R 777 /app

WORKDIR /app/libs/chatlas-agents

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHATLAS_AGENT_NAME=chatlas-agent

# Expose port if needed (for future API server)
EXPOSE 8000

# Entry point
ENTRYPOINT ["bash", "-c"]
CMD ["uv", "run", "chatlas", "--help"]
# CMD ["--help"]
