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

# Install the package
RUN cd /app/libs/chatlas-agents && \
    uv sync 

# Create a non-root user for security
RUN useradd -m -u 1000 chatlas && chown -R chatlas:chatlas /app
USER chatlas

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHATLAS_AGENT_NAME=chatlas-agent

# Expose port if needed (for future API server)
EXPOSE 8000

# Entry point
ENTRYPOINT ["uv", "run", "chatlas"]
CMD ["--help"]
