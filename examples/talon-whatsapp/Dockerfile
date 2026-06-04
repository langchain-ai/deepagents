FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_PATH=/opt/whatsapp-bridge/node_modules \
    PATH="/opt/deepagents-talon-venv/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/deepagents-talon-venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        ffmpeg \
        fonts-liberation \
        nodejs \
        npm && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY libs/deepagents ./libs/deepagents
COPY libs/code ./libs/code
COPY libs/talon ./libs/talon
COPY examples/talon-whatsapp/AGENTS.md /opt/deepagents-talon/AGENTS.md

RUN mkdir -p /opt/whatsapp-bridge && \
    cp libs/talon/deepagents_talon/channels/whatsapp_bridge/package.json \
        /opt/whatsapp-bridge/package.json && \
    cd /opt/whatsapp-bridge && \
    npm install --omit=dev

RUN cd libs/talon && uv sync --frozen --extra speech

CMD ["/opt/deepagents-talon-venv/bin/deepagents-talon", "--whatsapp"]
