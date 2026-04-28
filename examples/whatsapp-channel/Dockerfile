FROM node:20-slim AS node

FROM python:3.12-slim

# Copy Node.js runtime from the official image (includes npm/npx and all lib modules)
COPY --from=node /usr/local/bin/ /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules

# Install Chromium and system deps for whatsapp-web.js
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        curl \
        ffmpeg \
        fonts-liberation \
        libnss3 \
        libatk-bridge2.0-0 \
        libdrm2 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpango-1.0-0 \
        libcairo2 \
        libcups2 \
        libxss1 \
        libxtst6 \
        psmisc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Point puppeteer/whatsapp-web.js at the system Chromium
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV CHROME_PATH=/usr/bin/chromium

WORKDIR /app

# Install Node bridge deps first (cached layer)
COPY examples/whatsapp-channel/bridge/package.json examples/whatsapp-channel/bridge/package-lock.json* bridge/
RUN cd bridge && npm install --omit=dev

# Install deepagents from the local monorepo (version may not be on PyPI yet)
COPY libs/deepagents /tmp/libs/deepagents
RUN pip install --no-cache-dir /tmp/libs/deepagents && rm -rf /tmp/libs

# Install remaining Python deps
# Pass --build-arg INSTALL_SPEECH=true to include nemo_toolkit[asr] and
# pre-download the Parakeet TDT 0.6B v3 model into the image layer.
ARG INSTALL_SPEECH=false
COPY examples/whatsapp-channel/pyproject.toml .
RUN pip install --no-cache-dir -e . && \
    if [ "$INSTALL_SPEECH" = "true" ]; then \
        pip install --no-cache-dir 'nemo_toolkit[asr]>=2.4.0'; \
    fi

# Pre-download Parakeet TDT model at build time (baked into the image layer so
# containers start cold without re-downloading ~2.4 GB from HuggingFace).
RUN if [ "$INSTALL_SPEECH" = "true" ]; then \
        SPEECH_DEVICE=cpu TRANSFORMERS_OFFLINE=0 python -c \
        "import nemo.collections.asr as nemo_asr; \
         m = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); \
         print('Parakeet TDT 0.6B v3 model cached')"; \
    fi

# Copy application code (explicit files to avoid copying host session/media dirs
# — the .dockerignore in the example dir is ignored since the build context is repo root)
COPY examples/whatsapp-channel/*.py .
COPY examples/whatsapp-channel/cron/ cron/
COPY examples/whatsapp-channel/bridge/bridge.js bridge/
COPY examples/whatsapp-channel/.env.example .
COPY examples/whatsapp-channel/README.md .

# Session and media are stored on volumes
VOLUME ["/app/session", "/app/media"]

WORKDIR /app/workspace
CMD ["python", "../main.py"]
