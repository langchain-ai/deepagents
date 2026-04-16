FROM node:20-slim AS node

FROM python:3.12-slim

# Copy Node.js runtime from the official image (includes npm/npx and all lib modules)
COPY --from=node /usr/local/bin/ /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules

# Install Chromium and system deps for whatsapp-web.js
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
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

# Point puppeteer/whatsapp-web.js at the system Chromium
ENV PUPPETEER_SKIP_DOWNLOAD=true
ENV CHROME_PATH=/usr/bin/chromium

WORKDIR /app

# Install Node bridge deps first (cached layer)
COPY bridge/package.json bridge/package-lock.json* bridge/
RUN cd bridge && npm install --omit=dev

# Install Python deps (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Session and media are stored on volumes
VOLUME ["/app/session", "/app/media"]

CMD ["python", "main.py"]
