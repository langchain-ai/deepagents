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

# Install dependencies and build all packages in order (non-editable)
# This ensures the package is fully built during image creation
RUN cd /app/libs/deepagents && \
    uv venv /opt/venv && \
    uv pip install --no-cache . && \
    cd /app/libs/deepagents-cli && \
    uv pip install --no-cache . && \
    cd /app/libs/chatlas-agents && \
    uv pip install --no-cache .

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /app/libs/chatlas-agents

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CHATLAS_AGENT_NAME=chatlas-agent

# Expose port if needed (for future API server)
EXPOSE 8000

# Entry point - use the installed package directly
ENTRYPOINT ["chatlas"]
CMD ["--help"]
# CMD ["--help"]
