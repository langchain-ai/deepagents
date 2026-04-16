FROM python:3.12-slim

# Install Node.js 20, Chromium, and system deps in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
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
        fuser \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
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
